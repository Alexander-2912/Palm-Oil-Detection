# app.py
import os, io, tempfile, zipfile, warnings, hashlib
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import shape as shp_shape
import geopandas as gpd

# os, io, tempfile, zipfile, warnings, hashlib: utilitas sistem file, buffer memory, direktori sementara, 
# membuat arsip ZIP, menangani peringatan, hashing (dipakai untuk kunci cache).
# numpy as np: operasi numerik/array.
# streamlit as st: hanya untuk UI/caching model (kita abaikan UI, tapi st.cache_resource dipakai untuk caching model).
# PIL.Image: membuat/menyimpan PNG dari array.
# tensorflow dan load_model: memuat model Keras (U-Net ResNet*).
# rasterio (+ Window, Resampling): baca/tulis GeoTIFF/JP2 dan sampling ke window.
# rasterio.features.shapes: polygonize raster mask → geometri.
# shapely.geometry.shape (di-alias shp_shape): ubah GeoJSON geometry → Shapely polygon.
# geopandas as gpd: GeoDataFrame untuk menyimpan geometri dan atribut, serta menulis ke Shapefile.


# ====================== PARAMETER DEFAULT ======================
TILE, OVERLAP = 256, 64
# ukuran patch citra untuk dipotong (256) dan stride antar citra (64)
MULT, THRESH  = 32, 0.5
# kelipatan piksel (32) dan threshold untuk mask (0.5)
DISPLAY_BANDS = (1, 2, 3)
# band citra yang akan ditampilkan (1,2,3 = RGB)

PNG_MAX_SIDE  = 2000
# ukuran maksimal sisi citra preview PNG (2000 piksel)

BUILD_OVERVIEWS = True  
# tidak berpengaruh untuk PNG/preview, tapi dipakai saat tulis TIF


PIXEL_AREA_M2 = 100.0
# Baru: memaksa luas per piksel (m²). Jika bukan None, area dihitung dari angka ini 
# (contoh: Sentinel-2 resolusi 10 m → 10×10 = 100 m²/piksel). 
# Jika None, nanti dibaca dari geotransform raster.

# ====================== UTIL / HELPER ======================
# def scale_per_band_valid(arr, nodata, low=2, high=98):
#     # Mendefinisikan fungsi untuk menormalkan nilai citra per-band ke rentang [0, 1] memakai persentil.
#     x = arr.astype("float32")
#     # konversi array input ke float32 untuk presisi
#     if x.ndim == 2: x = x[..., None]
#     # Jika input 2D (H×W), tambahkan dimensi channel di belakang ⇒ (H×W×1).
#     if nodata is not None:
#         #Jika ada nilai NoData yang didefinisikan
#         for c in range(x.shape[-1]):
#         # Loop tiap channel c.
#             band = x[..., c]
#             # Ambil view band ke-c (band adalah view ke x[..., c]).
#             band[band == nodata] = np.nan
#             # Ganti semua piksel yang persis sama dengan nodata menjadi NaN.
#     out = np.empty_like(x)
#     # Inisialisasi array output dengan bentuk dan tipe yang sama seperti x.
#     for c in range(x.shape[-1]):
#     # Loop tiap channel c.
#         band = x[..., c]
#         # Ambil view band ke-c (band adalah view ke x[..., c]).
#         lo = np.nanpercentile(band, low)
#         hi = np.nanpercentile(band, high)
#         # Hitung persentil bawah/atas (mis. 2% & 98%) dengan mengabaikan NaN
#         out[..., c] = np.clip((band-lo)/(hi-lo+1e-6), 0, 1)
#         # band - lo = menggeser nilai citra supaya batas bawah (lo) menjadi nol.
#         # (hi - lo + 1e-6) = skala rentang nilai citra supaya batas atas (hi) menjadi 1.
#     return np.nan_to_num(out, nan=0.0)
#     # Mengembalikan array float32 bernilai di [0, 1] per band, tanpa NaN.

# ===== add near other helpers =====
# warna lembut (silakan ganti kalau mau)
COLOR_NONPALM = (220, 70, 60)    # merah lembut (kelas 0)
COLOR_PALM    = (0, 255, 0)  # hijau lembut (kelas 1)

def save_mask_png(mask_uint8, out_png, transparent_value=255):
    """
    mask_uint8: array 2D berisi {0=non-sawit, 1=sawit, 255=NoData}
    out_png   : path file png tujuan
    """
    pal = [0]*768  # 256*3
    pal[0:3]   = COLOR_NONPALM          # index 0 -> merah lembut
    pal[3:6]   = COLOR_PALM             # index 1 -> hijau lembut
    # sisanya biarkan hitam

    img = Image.fromarray(mask_uint8.astype("uint8"), mode="P")
    img.putpalette(pal)
    # bikin nilai 255 transparan (area di luar citra)
    if transparent_value is not None:
        img.info["transparency"] = transparent_value
    img.save(out_png, optimize=True)


def pad_to_multiple(img, multiple=32):
    # Mendefinisikan fungsi untuk menambah padding di bawah & kanan citra 
    # supaya tinggi dan lebar menjadi kelipatan multiple
    H,W = img.shape[:2]
    # Ambil tinggi (H) dan lebar (W) dari img.
    new_h = int(np.ceil(H/multiple)*multiple)
    new_w = int(np.ceil(W/multiple)*multiple)
    # Hitung target tinggi/lebar terdekat ke atas yang merupakan kelipatan multiple
    # Contoh: H=245, multiple=32 → 245/32=7.656…, ceil=8 → new_h=8*32=256.
    pad_h, pad_w = new_h-H, new_w-W
    # Hitung berapa banyak piksel yang perlu ditambahkan di bawah (pad_h) dan kanan (pad_w).
    if pad_h==0 and pad_w==0: return img,(0,0)
    #Early return: kalau ukuran sudah kelipatan multiple, tidak perlu padding.
    # img_pad = np.pad(img, ((0,pad_h),(0,pw:=pad_w),(0,0)), mode="reflect")
    img_pad = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
    # Tinggi: (0, pad_h) → tambah di bawah saja.
    # Lebar: (0, pw:=pad_w) → tambah di kanan
    # mode="reflect" → nilai padding adalah refleksi dari tepi citra
    # return img_pad,(pad_h,pw) # coba diapus
    return img_pad, (pad_h, pad_w)

def pad_to_size(img, th, tw):
    # th, tw: target height dan width minimal.
    H,W = img.shape[:2]
    # Mengambil tinggi (H) dan lebar (W) dari dua dimensi pertama img.
    ph = max(0, th-H)
    pw = max(0, tw-W)
    # ph (pad height) = berapa baris yang perlu ditambahkan di bawah agar tinggi ≥ th.
    # pw (pad width) = berapa kolom yang perlu ditambahkan di kanan agar lebar ≥ tw.
    # H=240, th=256 → ph=16; W=270, tw=256 → pw=0.
    if ph==0 and pw==0: return img,(0,0)
    # Early exit: jika tinggi & lebar sudah memenuhi target (tidak perlu padding), langsung kembalikan:
    img_pad = np.pad(img, ((0,ph),(0,pw),(0,0)), mode="reflect")
    # Pola padding per dimensi: ((atas, bawah), (kiri, kanan), (sebelumC, sesudahC)).
    return img_pad,(ph,pw)

def unpad(img, pad_hw):
    ph,pw = pad_hw
    # pad_hw: tuple (ph, pw) = jumlah padding yang dulu ditambahkan di bawah (ph) dan kanan (pw).
    if ph>0: img = img[:-ph,:,:]
    # ambil dari awal sampai sebelum ph elemen terakhir
    if pw>0: img = img[:,:-pw,:]
    # # ambil dari awal sampai sebelum ph elemen terakhir
    return img

# ---- Global percentile + gamma untuk visual ----
def compute_global_minmax_percentile(path, bands=(1,2,3), low=2, high=98, sample_step=64):
    # citra di scan dengan step 64 per band untuk menghitung persentil global
    lows, highs = [], []
    # inisialisasi nilai persentil atas dan bawah 
    with rasterio.open(path) as src:
    # membuka raster dengan rasterio dari temp dir
        bands = tuple(b for b in bands if 1 <= b <= src.count) or (1,)
        # Ambil setiap b dari bands, hanya jika b berada di antara 1 dan src.count
        # src.count: jumlah band pada raster
        # ngambil band yang valid saja (1, src.count)
        nodata = src.nodata
        # Ambil nilai NoData dari metadata raster (bisa None jika tidak didefinisikan).
        for b in bands:
        # Mulai loop per band yang sudah tervalidasi.
            vals = []
            # Siapkan penampung sementara vals (list of arrays) untuk menyimpan sampel nilai dari band b pada berbagai window.
            for row in range(0, src.height, sample_step):
                # Loop baris dengan lompatan sample_step
                # src.height = jumlah baris piksel (tinggi raster).
                h = min(sample_step, src.height - row)
                # berfungsi untuk menghindari membaca di luar batas gambar (out of range) saat memproses potongan raster di tepi bawah citra.
                # ngebandingin sisa citra (src.height - row) dengan sample_step, ambil yang lebih kecil.
                for col in range(0, src.width, sample_step):    
                # sama kayak atas, tapi untuk kolom (lebar raster)
                    w = min(sample_step, src.width - col)
                    # sama kayak atas, tapi untuk lebar
                    arr = src.read(b, window=Window(col, row, w, h)).reshape(-1)
                    # membaca data raster untuk band b pada window yang ditentukan (dari (col,row) dengan ukuran (w,h)).
                    # hasilnya di-flatten jadi array 1D dengan reshape(-1) berukuran 64.
                    if nodata is not None:
                        arr = arr[arr != nodata]
                        # Jika raster mendefinisikan NoData, buang semua nilai yang sama dengan nodata.
                    if arr.size:
                        # memastikan hanya window yang punya nilai valid disimpan.
                        vals.append(arr.astype(np.float32, copy=False))
                        # simpan array nilai valid ke vals, dengan tipe float32 untuk efisiensi memori. kalau udah float32, gausah di copy ulang
            if vals:
            # Jika ada nilai valid yang terkumpul untuk band ini:
                v = np.concatenate(vals)
                # gabungkan semua array di vals menjadi satu array besar v.
                lows.append(float(np.percentile(v, low)))
                highs.append(float(np.percentile(v, high)))
                # hitung persentil low dan high dari v, simpan ke lows dan highs.
            else:
                lows.append(0.0); highs.append(1.0)
                # jika ga ada nilai valid, pakai default 0.0 dan 1.0
    return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)
    # outputnya berupa array float32 untuk lows dan highs per band (masing masing ada 3)

def scale_fixed_minmax(arr, lows, highs):
    a = arr.astype(np.float32, copy=False)
    if a.ndim == 2:
        lo, hi = float(lows[0]), float(highs[0])
        out = (a - lo) / (hi - lo + 1e-6)
        return np.clip(out, 0.0, 1.0)[..., None]
    C = a.shape[-1]
    lo = np.array(lows[:C], dtype=np.float32).reshape(1,1,C)
    hi = np.array(highs[:C], dtype=np.float32).reshape(1,1,C)
    out = (a - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0.0, 1.0)

def apply_gamma(img01, gamma=1.5):
    return np.clip(img01, 0, 1) ** (1.0 / gamma)

# ---- luas piksel dan hitung area ----
def pixel_area_m2_from_tif(tif_path: str, PIXEL_AREA_M2) -> float:
    with rasterio.open(tif_path) as ds:
    # buka raster dengan rasterio
        a = float(ds.transform.a) # ukuran piksel di arah x
        e = float(ds.transform.e) # ukuran pixel di arah y
        # ambil elemen a (skala x) dan e (skala y) dari affine transform
        # affine(a, b, c / d, e, f) -> a(x) dan e(y) itu skala piksel di x (lebar) dan y (tinggi)
        # c dan f adalah koordinat x dan y
        # b dan d nol karena tidak ada rotasi

        # a untuk lebar
        # e untuk tinggi

        area = abs(a * e)
        # hitung luas piksel (m²) sebagai |a × e|

        if not np.isfinite(area) or area <= 0:
            # jika area tidak valid (NaN, Inf, ≤0), pakai nilai paksa
            area = float(PIXEL_AREA_M2)
        return area

def count_pixels_eq1(tif_path: str) -> int:
    with rasterio.open(tif_path) as ds:
        # membuka file raster dengan rasterio
        arr = ds.read(1)
        # membaca band 1 ke dalam array NumPy
        return int((arr == 1).sum())
        # menghitung jumlah piksel dengan nilai 1 dan mengembalikannya sebagai integer

def fmt_area_triple(m2: float) -> str:
    ha  = m2 / 10_000.0
    km2 = m2 / 1_000_000.0
    return f"{m2:,.2f} m²  |  {ha:,.2f} ha  |  {km2:,.4f} km²"

# ---- TF: batasi alokasi VRAM (opsional) ----
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass

# ---- Cache model agar tidak load berulang ----
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    # memuat model Keras dari path dengan caching
    return load_model(model_path, compile=False)

# ---- Prediksi windowed ----
def predict_full_raster(raster_path, model, in_ch=3, tile=256, overlap=64, multiple=32, disp_bands=(1,2,3), global_lows=None, global_highs=None):
    assert tile % multiple == 0 and 0 <= overlap < tile
    assert global_lows is not None and global_highs is not None
    # pastikan tile kelipatan multiple, dan overlap tidak negatid dan lebih kecil dari tile

    with rasterio.open(raster_path) as src:
        # Buka raster dengan rasterio
        H,W    = src.height, src.width
        # tinggi dan lebar raster
        nodata = src.nodata
        # ambil nilai nodata dari metadata raster (bisa None)
        out    = np.zeros((H,W,1), dtype=np.float32)
        # inisialisasi output prediksi probabilitas (H×W×1)
        weight = np.zeros((H,W,1), dtype=np.float32)
        # untuk menghitung berapa kali tiap piksel diisi (untuk blending rata-rata di area overlap).
        idxs = list(disp_bands) if (in_ch==3 and src.count>=3) else [1]
        # tentukan indeks band yang akan dibaca dari raster
        #Jika model butuh 3 kanal dan raster memiliki ≥3 band → pakai disp_bands (default RGB: 1,2,3).
        stride = tile-overlap
        # Hitung langkah geser antar patch (stride). Contoh: 256−64 = 192.
        ys = range(0, H, stride); xs = range(0, W, stride)
        # Buat daftar koordinat y dan x untuk memotong patch (window) dari raster.
        for y in ys:
            # loop per y
            for x in xs:
                # loop per x
                y2=min(y+tile,H); x2=min(x+tile,W)
                # tentukan batas bawah patch (tidak boleh melebihi dimensi raster)
                # posisi akhir dari patch: y2 = y + tile, tapi dibatasi maksimal H
                win = Window.from_slices((y,y2),(x,x2))
                # Window adalah objek bawaan rasterio yang digunakan untuk menunjukkan area kecil
                # Window.from_slices((row_start, row_stop), (col_start,col_stop))
                if in_ch==3 and src.count>=3:
                    patch = src.read(indexes=idxs, window=win).transpose(1,2,0)
                    # indexes=idxs menentukan band mana yang diambil.
                    # window=win artinya hanya baca area kecil sesuai Window
                    # (shape) = (C, H, W) → transpose ke (H, W, C)
                else:
                    patch = src.read(1, window=win)[...,None]
                    # baca band 1 saja, shape (H, W) → tambahkan axis jadi (H, W, 1)

                if nodata is not None and np.all(patch==nodata):
                    # jika seluruh patch adalah nodata, lewati prediksi
                    # cara mengisi placeholder agar hasil akhir tetap punya nilai valid, meskipun patch ini tidak diproses model.
                    hh=y2-y; ww=x2-x
                    # hitung ukuran patch sebenarnya (bisa kurang dari tile di tepi citra)
                    out[y:y+hh, x:x+ww,:]+=0.0
                    # tambahkan nilai 0.0 ke setiap elemen pada area [y:y+hh, x:x+ww, :] di array out.
                    weight[y:y+hh, x:x+ww,:]+=1.0
                    # tambahkan nilai 1.0 ke setiap elemen di area [y:y+hh, x:x+ww, :] dari array weight.
                    continue

                # lewati nodata → 0 dulu agar tidak ganggu scaling
                if nodata is not None:
                    patch = patch.astype(np.float32)
                    patch[patch == nodata] = 0.0

                patch01 = scale_fixed_minmax(patch, global_lows, global_highs)
                # skala patch ke 0-1 per band dengan persentil 2-98
                patch_pad, _  = pad_to_multiple(patch01, multiple=multiple)
                # pad patch agar kelipatan multiple
                patch_pad, phw= pad_to_size(patch_pad, tile, tile)
                # pad patch agar sesuai ukuran tile (jika kurang di tepi citra)
                #patch_pad itu berisi 256×256 piksel dengan 3 channel (RGB).
                pred_pad = model.predict(patch_pad[None,...], verbose=0)[0]
                # model deep learning (seperti U-Net, CNN, dll) tidak menerima 3D input, melainkan 4D tensor
                # (batch_size, height, width, channels)
                # setelah [None, ...] ---> (1, 256, 256, 3)
                # “membungkus” satu gambar ke dalam batch berisi 1 gambar.
                # Dengan [0], “menghilangkan” dimensi batch (yaitu angka 1 di depan).
                pred     = unpad(pred_pad, phw)
                # Unpad prediksi, menghapus padding ekstra yang ditambahkan tadi, kembali ke ukuran patch sebelum pad_to_size.
                hh=min(pred.shape[0], y2-y); ww=min(pred.shape[1], x2-x)
                # Hitung ukuran patch sebenarnya (bisa kurang dari tile di tepi citra)
                pred = pred[:hh,:ww,:]
                # Slicing [:hh, :ww, :] mengambil baris 0..hh-1 dan kolom 0..ww-1 untuk semua channel.
                out[y:y+hh, x:x+ww,:]    += pred
                # Menjumlahkan (bukan menimpa) patch prediksi ke kanvas global out pada lokasi yang sesuai.
                weight[y:y+hh, x:x+ww,:] += 1.0
                # Menambah bobot 1.0 untuk setiap piksel dalam area patch pada kanvas weight.
        prob = out/np.maximum(weight,1e-6)
        # Jadi piksel yang diliput k kali akan dibagi k, menghasilkan nilai rata-rata prediksi di piksel itu → 
        # blending yang halus, menghilangkan seam antar patch.
        return prob

# ---- Tulis mask GeoTIFF mengikuti metadata referensi ----
def save_mask_geotiff_like(ref_path, mask01, out_path, nodata=255, compress="lzw"):
    if mask01.ndim == 3:
        mask01 = mask01[..., 0]
    mask01 = mask01.astype(np.uint8)

    with rasterio.open(ref_path) as src:
        crs = src.crs
        transform = src.transform
        height, width = mask01.shape

    prof = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": crs,
        "transform": transform,
        "compress": compress,
        "BIGTIFF": "IF_SAFER",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "pixel",
        "photometric": "MINISBLACK",
        "nodata": nodata,       # <-- 255
    }

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(mask01, 1)
        # 0=non-sawit (MERAH), 1=sawit (KUNING), 255=NoData (transparan)
        try:
            dst.write_colormap(1, {
                0:   (220,  70,  60, 255),
                1:   (0, 255, 0, 255),
                255: (0,    0,   0,   0),  # transparan
            })
        except Exception:
            pass

    if BUILD_OVERVIEWS:
        with rasterio.open(out_path, "r+") as dst:
            dst.build_overviews([2, 4, 8, 16, 32], resampling=Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")



# ---- Visual PNG (citra asli & overlay) ----
def _vis_global_gamma(a, lows, highs, gamma=1.5):
    # a array citra input
    vis01 = scale_fixed_minmax(a, lows, highs)
    # skala piksel ke 0-1 menggunakan low dan high (percentil)
    # if vis01.shape[-1] == 1:
        # cek jumlah channel
        # vis01 = np.repeat(vis01, 3, axis=-1)
        # jika 1 channel, ulangi jadi 3 channel (grayscale ke RGB)
    return apply_gamma(vis01, gamma=gamma)

def save_png_preview_from_tif(tif_path, out_png, lows, highs, max_side=2000):
    with rasterio.open(tif_path) as src:
        # membuka raster
        H, W = src.height, src.width
        # ambil dimensi tinggi dan lebar dari raster
        scale = max(H/max_side, W/max_side, 1.0)
        # digunakan untuk mengecilkan gambar besar agar tidak melebihi max_side
        # ambil yang terbesar dari ketiganya
        dH, dW = int(H/scale), int(W/scale)
        #dh, dw ukuran baru pake scale
        idxs = [1,2,3] if src.count >= 3 else [1]
        # tentukan band yang akan dibaca (RGB jika ada ≥3 band, else band 1)
        a = src.read(indexes=idxs, out_shape=(len(idxs), dH, dW),
                     resampling=Resampling.bilinear).transpose(1,2,0)
        # membaca band yang dipilih dengan ukuran diubah ke (dH, dW) menggunakan resampling bilinear > (C, dH, dW) → (dH, dW, C)
        vis01 = _vis_global_gamma(a, lows, highs, gamma=1.5)
        Image.fromarray((np.clip(vis01,0,1)*255).astype(np.uint8), mode="RGB").save(out_png, optimize=True)
        # np.clip mastiin nilai tetap di 0-1, terus diubah jadi 0-255
        # optimize=True supaya PNG lebih kecil ukurannya

def save_overlay_png_from_tif(base_tif, mask_tif, out_png, lows, highs,
                              max_side=2000, color_fg=(0, 255, 0), color_bg=(255, 0, 0), alpha=0.65):
    # color_fg buat warna overlay mask == 1 (default hijau)
    # color_bg buat warna overlay mask == 0 (default merah)
    # alpha buat transparansi
    with rasterio.open(base_tif) as src_b:
        Hb, Wb = src_b.height, src_b.width
        scale = max(Hb/max_side, Wb/max_side, 1.0)
        dH, dW = int(Hb/scale), int(Wb/scale)
        # sama kayak yang diatas

        idxs = [1,2,3]
        base_raw = src_b.read(indexes=idxs, out_shape=(len(idxs), dH, dW),
                              resampling=Resampling.bilinear).transpose(1,2,0)
        # baca band RGB dengan resampling bilinear ke ukuran (dH, dW), kemudian di tranpose ke (H, W, C)
        valid = np.any(base_raw != 0, axis=-1)
        # buat mask valid (bukan hitam semua di RGB)

        base01 = _vis_global_gamma(base_raw, lows, highs, gamma=1.5)
        # skala ke 0-1 dengan gamma untuk mencerahkan citra
        over   = (np.clip(base01, 0, 1) * 255.0).astype(np.float32)
        # klip ke 0-1 kemudian diubah ke 0-255 float32 untuk overlay

    with rasterio.open(mask_tif) as src_m:
        mask_small = src_m.read(1, out_shape=(dH, dW), resampling=Resampling.nearest).astype(np.uint8)
    # baca band 1 dari mask dan ubah ukuran dan esample ke (dH, dW) dengan resampling nearest (supaya nilai mask tetap utuh)

    rf,gf,bf = color_fg
    rb,gb,bb = color_bg
    # pecah band masking ke warna depan dan belakang
    a = float(alpha)
    # alpha sebagai float


    # ngewarnain dengan merah dan hijau
    m1 = (mask_small == 1) & valid
    # mask untuk piksel dengan mask == 1 dan valid (bukan area kosong)
    if m1.any():
        over[m1, 0] = (1 - a) * over[m1, 0] + a * rf
        over[m1, 1] = (1 - a) * over[m1, 1] + a * gf
        over[m1, 2] = (1 - a) * over[m1, 2] + a * bf
        # out = (1-α) * base + α * color

    m0 = (mask_small == 0) & valid
    # mask untuk piksel dengan mask == 0 dan valid
    if m0.any():
        over[m0, 0] = (1 - a) * over[m0, 0] + a * rb
        over[m0, 1] = (1 - a) * over[m0, 1] + a * gb
        over[m0, 2] = (1 - a) * over[m0, 2] + a * bb

    Image.fromarray(np.clip(over, 0, 255).astype(np.uint8), mode="RGB").save(out_png, optimize=True)
    # pastiin nilai 0-255, ubah ke uint8, simpan PNG

# ---- Polygonize mask menjadi Shapefile ----
def mask_tif_to_shapefile(mask_tif_path, shp_out_path, only_value=1, dissolve=False):
    with rasterio.open(mask_tif_path) as ds:
        arr = ds.read(1)
        # baca band 1 ke array 2D
        tr  = ds.transform
        # ambil affine transform untuk memetakan koordinat piksel
        # ambil elemen a (skala x) dan e (skala y) dari affine transform
        # affine(a, b, c / d, e, f) -> a(x) dan e(y) itu skala piksel di x (lebar) dan y (tinggi)
        # c dan f adalah koordinat x dan y
        # b dan d nol karena tidak ada rotasi
        crs = ds.crs
        # ambil CRS dari metadata raster (bisa None, mis. UTM EPSG:xxxxx)

    polys = [shp_shape(geom) for geom, val in shapes(arr, mask=(arr == only_value), transform=tr)
             if int(val) == int(only_value)]
    # shapes(arr, mask=..., transform=tr) menghasilkan generator (geom (geometri poligon dalam format json), val (nilai piksel))
    # hanya poligon dengan nilai piksel == only_value (true) yang diambil 
    # transform=tr untuk memetakan koordinat piksel ke koordinat dunia
    # shp_shape (geom) mengubah geometri GeoJSON ke Shapely polygon
    # if int(val) == int(only_value): filter tambahan untuk memastikan hanya nilai target yang disimpan (aman kalau arr bertipe float/uint)
    # Hasilnya: polys adalah list Shapely polygon untuk area bernilai only_value.

    gdf = gpd.GeoDataFrame({"value":[only_value]*len(polys)}, geometry=polys, crs=crs)
    # Buat GeoDataFrame dari daftar poligon.
    # kolom atribut value diisi only_value untuk setiap baris.
    # geometry=polys: set kolom geometri dari Shapely objects.
    # crs=crs: wariskan CRS raster agar shapefile tergeoreferensi.

    if gdf.empty:
        # Jika tidak ada piksel bernilai only_value → polys kosong → gdf.empty.
        gdf = gpd.GeoDataFrame({"value":[], "area_m2":[]}, geometry=[], crs=crs)
        # Buat GeoDataFrame kosong tapi dengan skema kolom yang diharapkan (value, area_m2) agar downstream aman.
        gdf.to_file(shp_out_path, driver="ESRI Shapefile")
        # Tulis Shapefile kosong (akan tetap membuat berkas .shp/.shx/.dbf/.prj dengan skema).
        return shp_out_path
        # Kembalikan path keluaran segera (tidak lanjut ke langkah hitung area/dissolve).

    gdf["area_m2"] = gdf.geometry.area.astype(float)
    # Hitung luas tiap poligon pada kolom geometry.
    # buat nyimpen koordinat dari setiap poligon dan ngitung luas dari masing-masing poligon yang berdampingan
    if dissolve:
        gdf = gdf.dissolve(by="value", as_index=False, aggfunc={"area_m2":"sum"})
        # gabungkan semua poligon dengan nilai atribut "value" yang sama menjadi satu poligon tunggal.
        # poligonnya ada banyak tapi dimasukkin ke satu file poligon
        # luas area_m2 dijumlahkan untuk poligon yang digabungkan.
    gdf.to_file(shp_out_path, driver="ESRI Shapefile")
    # shp menyimpan geometri
    # shx indeks posisi geometri
    # dbf tabel atribut kolom
    # prj sistem koordinat
    # cpg encoding karakter
    return shp_out_path
    

# ---- ZIP: mask TIF + shapefile + overlay + input asli ----
def zip_mask_and_shapefile(mask_tif_path: str, shp_path: str, overlay_png: str,
                           input_original: str, mask_png_path: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.isfile(mask_tif_path):
            zf.write(mask_tif_path, arcname=os.path.basename(mask_tif_path))
        if mask_png_path and os.path.isfile(mask_png_path):                      # <<< BARU
            zf.write(mask_png_path, arcname=os.path.basename(mask_png_path))     # <<< BARU
        stem, _ = os.path.splitext(shp_path)
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix", ".fix", ".xml"]:
            p = stem + ext
            if os.path.isfile(p):
                zf.write(p, arcname=os.path.basename(p))
        zf.write(overlay_png, arcname=os.path.basename(overlay_png))
        zf.write(input_original, arcname=os.path.basename(input_original))
    buf.seek(0)
    return buf.getvalue()








# ====================== UI / ROUTING ======================
st.set_page_config(page_title="Deteksi Sawit – U-Net ResNet", layout="wide")
if "results" not in st.session_state:
    st.session_state["results"] = []

if "hasil_nonce" not in st.session_state:
    st.session_state["hasil_nonce"] = 0

# gunakan query param untuk "routing" internal
qp = st.query_params
view = qp.get("view", "home")

if view != "hasil" and "results" in st.session_state:
    st.session_state.pop("results", None)

# ---------- SIDEBAR NAV (sesuai mockup) ----------
st.markdown(
    """
    <style>
    div.stButton > button {
        height: 80px;
        width: 100%;
        font-size: 22px;
        font-weight: 600;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Navigasi")
    def nav_btn(label, key_view):
        active = (view == key_view)
        if st.button(label, use_container_width=True, type="primary" if active else "secondary"):
            st.query_params["view"] = key_view
            st.rerun()
    nav_btn("Home", "home")
    nav_btn("Pengujian", "pengujian")
    nav_btn("User Manual", "manual")
    nav_btn("Tentang Pembuat", "pembuat")
    nav_btn("Tentang Model", "model")

# ---------- HALAMAN: HOME ----------
if view == "home":
    st.markdown(
        "<h2 style='text-align:center'>Deteksi Area Kebun Kelapa Sawit Pada Citra "
        "Sentinel-2 Menggunakan Semantic Segmentation U-Net Dengan Transfer Learning "
        "ResNet34</h2>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Layout tombol
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Pengujian", use_container_width=True):
            st.query_params["view"] = "pengujian"
            st.rerun()
    with c2:
        if st.button("User Manual", use_container_width=True):
            st.query_params["view"] = "manual"
            st.rerun()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("Tentang Pembuat", use_container_width=True):
            st.query_params["view"] = "pembuat"
            st.rerun()
    with c4:
        if st.button("Tentang Model", use_container_width=True):
            st.query_params["view"] = "model"
            st.rerun()

# ---------- HALAMAN: PENGUJIAN (inference) MULAI DARI SINI ----------
def run_pipeline(file_bytes: bytes, filename: str, model_path: str):

    
    # file disimpan ke temporary directory 
    work_dir = tempfile.mkdtemp(prefix="run_")
    in_path = os.path.join(work_dir, filename)
    with open(in_path, "wb") as f:
        f.write(file_bytes)

    # manggil fungsi compute_global_minmax_percentile untuk scaling visual
    GLOBAL_LOW, GLOBAL_HIGH = compute_global_minmax_percentile(
        in_path, bands=DISPLAY_BANDS, low=2, high=98, sample_step=64
    )

    #  load model dari path yang dipilih
    model = load_model_cached(model_path)
    #Lalu pipeline membaca model.inputs[0].shape[-1] untuk menentukan IN_CH:
    # Jika IN_CH==3 dan raster punya ≥3 band → tile input akan memakai band 1,2,3.
    # Jika tidak → tile input hanya dari band 1 (grayscale, shape H×W×1).
    try:
        IN_CH = int(model.inputs[0].shape[-1]) or 3
    except Exception:
        IN_CH = 3

    # prediksi full raster
    prob = predict_full_raster(
        in_path, model, in_ch=IN_CH,
        tile=TILE, overlap=OVERLAP, multiple=MULT, disp_bands=DISPLAY_BANDS, global_lows=GLOBAL_LOW, global_highs=GLOBAL_HIGH
    )
    mask = (prob[..., 0] >= THRESH).astype(np.uint8)

    # Buat valid mask dari citra input (alpha/bitmask internal)
    with rasterio.open(in_path) as src:
        valid_full = (src.read_masks(1) > 0)

    # Jadikan area di luar citra = 255 (NoData)
    mask_export = mask.copy()
    mask_export[~valid_full] = 255

    mask_tif_path = os.path.join(work_dir, "prediction_mask.tif")
    save_mask_geotiff_like(in_path, mask_export, mask_tif_path, nodata=255)

    mask_png_path = os.path.join(work_dir, "prediction_mask.png")
    save_mask_png(mask_export, mask_png_path, transparent_value=255)

    # ================== STATISTIK PIXEL (SAWIT & NON-SAWIT) ==================
    # Piksel sawit (mask == 1)
    n_pixels_pos = count_pixels_eq1(mask_tif_path)

    # Piksel valid = hanya area citra (bukan putih/kosong)
    # pakai read_masks seperti di skrip kedua kamu
    with rasterio.open(mask_tif_path) as ds_mask:
        valid_mask = ds_mask.read_masks(1) > 0
        n_pixels_valid = int(valid_mask.sum())

    # Piksel bukan kebun sawit (kelas 0 di area valid)
    n_pixels_nonpalm = max(n_pixels_valid - n_pixels_pos, 0)

    # Luas per piksel (dipaksa 100 m² seperti parameter global)
    px_area_m2 = float(PIXEL_AREA_M2)

    # Luas kebun sawit
    area_m2      = n_pixels_pos * px_area_m2
    area_ha      = area_m2 / 10_000.0
    area_km2     = area_m2 / 1_000_000.0

    # Luas bukan kebun sawit (hanya area valid citra)
    area_nonpalm_m2  = n_pixels_nonpalm * px_area_m2
    area_nonpalm_ha  = area_nonpalm_m2 / 10_000.0
    area_nonpalm_km2 = area_nonpalm_m2 / 1_000_000.0


    shp_path = os.path.join(work_dir, "prediction_mask.shp")
    mask_tif_to_shapefile(mask_tif_path, shp_path, only_value=1, dissolve=True)

    png_rgb_path = os.path.join(work_dir, "input_preview_rgb.png")
    # buat direktor baru untuk simpan preview PNG
    # cuma buat preview di streamlit doang
    png_ovr_path = os.path.join(work_dir, "input_overlay.png")
    # buat direktor baru untuk simpan overlay PNG
    save_png_preview_from_tif(in_path, png_rgb_path, GLOBAL_LOW, GLOBAL_HIGH, max_side=PNG_MAX_SIDE)
    save_overlay_png_from_tif(in_path, mask_tif_path, png_ovr_path, GLOBAL_LOW, GLOBAL_HIGH, max_side=PNG_MAX_SIDE)

    with open(png_rgb_path, "rb") as f: png_rgb_bytes = f.read()
    with open(png_ovr_path, "rb") as f: png_ovr_bytes = f.read()
    with open(mask_tif_path, "rb") as f: mask_tif_bytes = f.read()
    # mode rb supaya bener bacanya (binary)
    # f.read mengembalikan isi file sebagai bytes
    # hasilnya disimpan di variabel bytes

    # karena png itu file biner, bukan teks (r)
    # read dipake buat baca dari path (memori lokal) ke bytes (di ram), karena browser gaada akses ke path temporary

    zip_bytes = zip_mask_and_shapefile(mask_tif_path, shp_path,
                                       overlay_png=png_ovr_path, input_original=in_path, mask_png_path=mask_png_path)

    return {
        "work_dir": work_dir,
        "png_rgb_bytes": png_rgb_bytes,
        "png_ovr_bytes": png_ovr_bytes,
        "mask_tif_bytes": mask_tif_bytes,
        "zip_bytes": zip_bytes,
        "zip_name": f"{os.path.splitext(filename)[0]}_results.zip",
        "mask_name": "prediction_mask.tif",
        # (BARU) metrik luas kebun sawit
        "n_pixels_pos": n_pixels_pos,
        "pixel_area_m2": float(PIXEL_AREA_M2),
        "area_m2": area_m2,
        "area_ha": area_ha,
        "area_km2": area_km2,
        "fmt_area": fmt_area_triple(area_m2),
        # (BARU) metrik luas non-sawit di area valid
        "n_pixels_valid": n_pixels_valid,
        "n_pixels_nonpalm": n_pixels_nonpalm,
        "area_nonpalm_m2": area_nonpalm_m2,
        "area_nonpalm_ha": area_nonpalm_ha,
        "area_nonpalm_km2": area_nonpalm_km2,
        "filename": filename,
    }




if view == "pengujian":
    st.title("Pengujian")
    st.divider()

    # pilih model via dua tombol (single select)
    st.subheader("Model")
    if "model_sel" not in st.session_state:
        st.session_state.model_sel = "models/res18_backbone_100epochs_256x192_compiled.hdf5"
    c1, c2 = st.columns(2)
    if c1.button("ResNet18", type="primary" if st.session_state.model_sel.endswith("res18_backbone_100epochs_256x192_compiled.hdf5") else "secondary"):
        st.session_state.model_sel = "models/res18_backbone_100epochs_256x192_compiled.hdf5"
    if c2.button("ResNet34", type="primary" if st.session_state.model_sel.endswith("res34_backbone_100epochs_256x192_compiled.hdf5") else "secondary"):
        st.session_state.model_sel = "models/res34_backbone_100epochs_256x192_compiled.hdf5"

    year_peng = st.text_input("Tahun", placeholder="mis. 2019", key="year_pengujian")  # BARU

    st.subheader("Upload citra (TIF/JP2)")
    uploaded = st.file_uploader(" ", type=["tif","tiff","jp2"], accept_multiple_files=False, label_visibility="collapsed")
    # if uploaded is not None:
    #     new_key = hashlib.md5(uploaded.getbuffer()).hexdigest() + "|" + st.session_state.model_sel
    #     # membuat hash dari file dan  nama model
    #     if st.session_state.get("last_key") != new_key:
    #         st.session_state.pop("result", None)
    #         st.session_state["last_key"] = new_key

    process = st.button("Proses", use_container_width=True, type="primary")

    if process:
        if uploaded is None:
            st.error("Silakan upload 1 file citra terlebih dahulu.")
            st.stop()
        with st.spinner("Inferensi berjalan..."):
            res_new = run_pipeline(uploaded.getbuffer(), uploaded.name, st.session_state.model_sel)
            yr = (st.session_state.get("year_pengujian", "") or "").strip()            # BARU
            res_new["year"] = int(yr) if yr.isdigit() else None
            # siapkan list results
            if "results" not in st.session_state or not isinstance(st.session_state["results"], list):
                st.session_state["results"] = []
            st.session_state["results"].append(res_new)
        st.success("Selesai diproses.")
        st.query_params["view"] = "hasil"
        st.rerun()

# ---------- HALAMAN: HASIL  ----------
elif view == "hasil":
    if "results" not in st.session_state:
        st.session_state["results"] = []
    st.markdown("<h2 style='text-align:center'>Hasil Pengujian</h2>", unsafe_allow_html=True)
    st.divider()

    results = st.session_state.get("results", [])
    if not results:
        st.warning("Belum ada hasil. Buka menu Pengujian untuk memproses citra.")
        if st.button("⬅️ Ke Pengujian", use_container_width=True):
            st.query_params["view"] = "pengujian"
            st.rerun()
        st.stop()

    # ================== FORM PENGUJIAN (SAMA PERSIS, SEKARANG DI HALAMAN HASIL) ==================
    st.subheader("Pengujian")

    # Model selector (pakai state global yang sama)
    if "model_sel" not in st.session_state:
        st.session_state.model_sel = "models/res18_backbone_100epochs_256x192_compiled.hdf5"

    cc1, cc2 = st.columns(2)
    if cc1.button("ResNet18", key="hasil_res18_btn",
                  type="primary" if st.session_state.model_sel.endswith("res18_backbone_100epochs_256x192_compiled.hdf5") else "secondary",
                  use_container_width=True):
        st.session_state.model_sel = "models/res18_backbone_100epochs_256x192_compiled.hdf5"
    if cc2.button("ResNet34", key="hasil_res34_btn",
                  type="primary" if st.session_state.model_sel.endswith("res34_backbone_100epochs_256x192_compiled.hdf5") else "secondary",
                  use_container_width=True):
        st.session_state.model_sel = "models/res34_backbone_100epochs_256x192_compiled.hdf5"

    year_hasil = st.text_input("Tahun", placeholder="mis. 2021",
                           key=f"year_hasil_{st.session_state.hasil_nonce}") 
    uploaded2 = st.file_uploader("Upload citra (TIF/JP2)", type=["tif","tiff","jp2"],
                             accept_multiple_files=False,
                             key=f"uploader_hasil_{st.session_state.hasil_nonce}")
    process2 = st.button("Proses Pengujian Baru", type="primary", use_container_width=True, key="process_hasil")

    if process2:
        if uploaded2 is None:
            st.error("Silakan upload 1 file citra terlebih dahulu.")
            st.stop()
        with st.spinner("Inferensi berjalan..."):
            res_new = run_pipeline(uploaded2.getbuffer(), uploaded2.name, st.session_state.model_sel)
            yr2 = (st.session_state.get("year_hasil", "") or "").strip()                # BARU
            res_new["year"] = int(yr2) if yr2.isdigit() else None                       # BARU
            yr2 = (year_hasil or "").strip()                          # BARU
        res_new["year"] = yr2 if yr2 else None                    # BARU
        st.session_state["results"].append(res_new)
        # BARU: reset field dengan memaksa remount widget
        st.session_state["hasil_nonce"] += 1                      # BARU
        st.success("Pengujian baru selesai.")
        st.rerun()

    # === Grafik garis dinamis berdasarkan TAHUN (string) ===  # BARU (perbaikan)
    st.subheader("Perbandingan Luas Kebun (km²) per Tahun")

    # Kumpulkan (year, area) dari semua hasil yang punya year
    rows = []
    for r in results:
        y = r.get("year")
        if y is None or str(y).strip() == "":
            continue
        try:
            y_int = int(str(y).strip())
        except Exception:
            continue
        rows.append({"Tahun": y_int, "Luas (km²)": float(r["area_km2"])})

    if rows:
        df = pd.DataFrame(rows)

        # Jika ada beberapa hasil pada tahun yang sama, kita jumlahkan (total luas per tahun)
        df_year = (df.groupby("Tahun", as_index=True)["Luas (km²)"]
                    .sum()
                    .sort_index()
                    .reset_index())

        # Grafik garis
        df_plot = df_year.copy()
        df_plot["Tahun"] = df_plot["Tahun"].astype(int).astype(str)

        # Jadikan Tahun sebagai index (kategori diskrit), lalu plot
        df_plot = df_plot.set_index("Tahun")
        st.line_chart(df_plot["Luas (km²)"])

        # Hitung persentase perubahan year-to-year
        # %Δ = (tahun_i - tahun_(i-1)) / tahun_(i-1) * 100
        df_year["%Δ vs tahun sebelumnya"] = df_year["Luas (km²)"].pct_change() * 100.0

        # Buat tabel pasangan periode, mis. 2023→2024, 2024→2025
        pairs = []
        for i in range(1, len(df_year)):
            y_prev = int(df_year.loc[i-1, "Tahun"])
            y_curr = int(df_year.loc[i,   "Tahun"])
            v_prev = float(df_year.loc[i-1, "Luas (km²)"])
            v_curr = float(df_year.loc[i,   "Luas (km²)"])
            if v_prev == 0:
                pct = None
            else:
                pct = (v_curr - v_prev) / v_prev * 100.0
            pairs.append({
                "Periode": f"{y_prev}→{y_curr}",
                "Luas Tahun Awal (km²)": v_prev,
                "Luas Tahun Akhir (km²)": v_curr,
                "Persentase Perubahan": pct
            })

        # Tabel persentase perubahan antar tahun
        if pairs:
            df_pairs = pd.DataFrame(pairs)

            # Format tampilan persentase agar rapi (+/− dan 2 desimal)
            def fmt_pct(x):
                return "—" if x is None or pd.isna(x) else f"{x:+.2f}%"

            df_pairs_display = df_pairs.copy()
            df_pairs_display["Persentase Perubahan"] = df_pairs_display["Persentase Perubahan"].apply(fmt_pct)
            st.subheader("Persentase Perubahan Antar Tahun")
            st.dataframe(df_pairs_display, use_container_width=True)

            # Perubahan total dari tahun pertama ke terakhir (mis. 2023→2025)
            first_year = int(df_year.iloc[0]["Tahun"])
            last_year  = int(df_year.iloc[-1]["Tahun"])
            first_val  = float(df_year.iloc[0]["Luas (km²)"])
            last_val   = float(df_year.iloc[-1]["Luas (km²)"])
            if first_val == 0:
                total_pct = None
            else:
                total_pct = (last_val - first_val) / first_val * 100.0

            abs_change = last_val - first_val

            # Tampilkan metric ringkas
            st.metric(
                label=f"Perubahan Total {first_year}→{last_year}",
                value=f"{abs_change:.4f} km²",
                delta=("—" if total_pct is None else f"{total_pct:+.2f}%")
            )

        else:
            st.info("Butuh minimal dua tahun berbeda untuk menghitung persentase perubahan.")

    else:
        st.info("Masukkan tahun pada form agar grafik dan persentase dapat ditampilkan.")




    st.divider()

    # ================== TAMPILKAN SEMUA HASIL PENGUJIAN ==================
    for idx, res in enumerate(results, start=1):
        st.markdown(f"### Pengujian {idx} — *{res.get('filename','(tanpa nama)')}*")
        cA, cB = st.columns(2)
        with cA:
            st.subheader("Citra asli (preview)")
            st.image(res["png_rgb_bytes"], use_container_width=True)
        with cB:
            st.subheader("Overlay")
            st.image(res["png_ovr_bytes"], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Merah = bukan area kebun kelapa sawit")
        with col2:
            st.caption("Hijau = area kebun kelapa sawit")

        # metrik luas untuk masing-masing pengujian
        st.subheader("Luas Area Kebun Kelapa Sawit Terdeteksi")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah piksel positif", f"{res['n_pixels_pos']:,}")
        m2.metric("Luas/piksel", f"{res['pixel_area_m2']:.2f} m²")
        m3.metric("Luas total (ha)", f"{res['area_ha']:.2f} ha")
        m4.metric("Luas total (km²)", f"{res['area_km2']:.4f} km²")

        st.caption(f"≈ {res['area_m2']:,.2f} m² total (dihitung dari {res['n_pixels_pos']:,} piksel × {res['pixel_area_m2']:.2f} m²/piksel).")

                # ================== BARIS BARU: BUKAN KEBUN SAWIT ==================
        st.subheader("Luas Area BUKAN Kebun Kelapa Sawit")
        n_np = res.get("n_pixels_nonpalm", None)

        if n_np is not None:
            n_valid = res.get("n_pixels_valid", n_np + res["n_pixels_pos"])
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Jumlah piksel negatif", f"{n_np:,}")
            m6.metric("Luas/piksel", f"{res['pixel_area_m2']:.2f} m²")
            m7.metric("Luas total (ha)", f"{res['area_nonpalm_ha']:.2f} ha")
            m8.metric("Luas total (km²)", f"{res['area_nonpalm_km2']:.4f} km²")

            st.caption(
                f"≈ {res['area_nonpalm_m2']:,.2f} m² total "
                f"(dihitung dari {n_np:,} piksel × {res['pixel_area_m2']:.2f} m²/piksel"
            )
        else:
            st.info("Statistik area bukan kebun sawit tidak tersedia untuk hasil ini.")


        # tombol unduh per pengujian
        st.download_button(
            "Unduh hasil (mask tif + shapefile + overlay PNG + input asli)",
            data=res["zip_bytes"],
            file_name=res["zip_name"],
            mime="application/zip",
            use_container_width=True,
            key=f"dl_zip_{idx}"
        )
        st.divider()

    # navigasi kembali
    if st.button("⬅️ Ke Pengujian", use_container_width=True, key="back_to_pengujian_from_hasil"):
        st.query_params["view"] = "pengujian"
        st.rerun()


# ---------- HALAMAN: USER MANUAL ----------
elif view == "manual":
    st.markdown("<h2 style='text-align:center'>User Manual</h2>", unsafe_allow_html=True)
    st.divider()

    st.subheader("Halaman Utama")
    st.info("""
    Pada halaman home, pengguna akan ditampilkan sebuah *side navigation bar* yang dapat digunakan untuk mengakses halaman-halaman lainnya. 
    Pada tampilan utama halaman home, pengguna akan ditampilkan judul dari perancangan yang dibuat, yaitu **“Deteksi Area Kebun Kelapa Sawit pada Citra Sentinel-2 Menggunakan Semantic Segmentation U-Net Dengan Transfer Learning ResNet34”**. 
    Pada halaman ini juga terdapat empat buah tombol yang dapat digunakan untuk bernavigasi ke halaman lainnya, yaitu:

    1. **Pengujian**  
    Tombol ini digunakan untuk memindahkan pengguna ke halaman pengujian.

    2. **User Manual**  
    Tombol ini digunakan untuk memindahkan pengguna ke halaman user manual.

    3. **Tentang Pembuat**  
    Tombol ini digunakan untuk memindahkan pengguna ke halaman tentang pembuat.

    4. **Tentang Model**  
    Tombol ini digunakan untuk memindahkan pengguna ke halaman tentang model.
    """)

    st.subheader("Halaman Pengujian")
    st.info("""
    Pada halaman pengujian, pengguna akan diberikan sebuah form yang berfungsi untuk memilih dan memasukkan data. 
            
    1.	Model
    Pada bagian model, pengguna akan diberikan dua buah pilihan, yaitu ResNet18 dan ResNet34. Apabila pengguna menekan tombol ResNet18, maka pengguna memilih untuk menggunakan model U-Net dengan ResNet18 untuk melakukan pengujian. Namun, apabila pengguna menekan tombol ResNet34, maka pengguna memilih untuk menggunakan model U-Net dengan ResNet34 untuk melakukan pengujian.
            
    2.	Input Data
    Pada bagian input data, pengguna akan diminta untuk menginput data yang ingin dilakukan pengujian. Data yang diinput harus berupa citra Satelit Sentinel-2 dengan ekstensi data yang diperbolehkan adalah JP2 atau tiff.
            
    3.	Hasil
    Tombol hasil digunakan untuk memulai proses pengujian menggunakan model yang telah dipilih dan input data yang dimasukkan. Tombol hasil juga akan memindahkan pengguna ke halaman hasil untuk menunjukkan hasil dari pengujian yang dilakukan
    """)

    st.subheader("Halaman Hasil Pengujian")
    st.info("""Pada halaman hasil pengujian, pengguna akan ditampilkan hasil dari pengujian yang dilakukan. 
            Hasil yang ditampilkan adalah gambar input data yang dimasukkan beserta hasil pengujian deteksi area kebun 
            kelapa sawit menggunakan model yang telah dipilih. Kedua gambar akan diletakkan bersampingan untuk mempermudah 
            melihat perbedaan antara input data dengan hasil pengujian yang telah memberikan label pada area-area kebun kelapa 
            sawit. """)

    st.subheader("Halaman Tentang Pembuat")
    st.info("""Pada halaman tentang pembuat, pengguna akan ditampilkan dua bagian penjelasan utama, yaitu deskripsi pembuat dan 
            latar belakang perancangan. Pada bagian deskripsi pembuat, terdapat penjelasan mengenai pembuat yang melakukan 
            perancangan ini, seperti biodata dan penjelasan lainnya mengenai pembuat. Pada bagian latar belakang rancangan, 
            terdapat penjelasan mengenai latar belakang alasan pembuatan rancangan ini.""")
    
    
    st.subheader("Halaman Tentang Model")
    st.info("""Pada halaman tentang model, pengguna akan ditampilkan dua bagian penjelasan utama, yaitu deskripsi model U-Net 
            dengan ResNet18 dan deskripsi model U-Net dengan ResNet34. Masing-masing bagian akan menjelaskan mengenai model yang 
            digunakan. Penjelasan yang diberikan seperti metrik evaluasi dari training model tersebut.""")
    
    st.subheader("Halaman User Manual")
    st.info("""Pada halaman user manual, pengguna akan ditampilkan penjelasan mengenai isi dan langkah-langkah yang dapat 
            dilakukan pada masing-masing halaman. Halaman user manual akan memberikan enam bagian penjelasan utama yang 
            masing-masing untuk satu halaman pada perancangan yang dibuat, yaitu halaman home, halaman pengujian, halaman hasil, 
            halaman tentang pembuat, dan halaman tentang model.""")

# ---------- HALAMAN: TENTANG PEMBUAT ----------
elif view == "pembuat":
    st.markdown("<h2 style='text-align:center'>Deskripsi Pembuat</h2>", unsafe_allow_html=True)
    st.divider()

    # --- 2 kolom: kiri foto, kanan biodata ---
    col_foto, col_bio = st.columns([1, 2], gap="large")

    with col_foto:
        st.image("images/foto_diri.jpg", caption="Vincent Alexander", use_container_width=True)

    with col_bio:
        st.markdown(
            """
    **Vincent Alexander** — NIM 535220149  
    Mahasiswa yang membuat perancangan *semantic segmentation* untuk deteksi area kebun kelapa sawit pada citra Sentinel-2 menggunakan arsitektur **U-Net** dengan *transfer learning* **ResNet18/ResNet34**.

    **Biodata singkat**  
    - Program Studi: Teknik Informatika 
    - Minat: Computer Vision, Remote Sensing, Deep Learning  
    - Topik Skripsi: Deteksi Area Kebun Kelapa Sawit Pada Citra Sentinel-2 Menggunakan Semantic Segmentation U-Net Dengan Transfer Learning ResNet34  
    - Email: alex.535220149@stu.untar.ac.id""")
        
    st.divider()
    
    st.subheader("Latar Belakang Perancangan")
    st.info("""
    Aplikasi Streamlit ini dirancang sebagai media interaktif untuk menampilkan hasil penelitian mengenai deteksi dan 
    segmentasi area perkebunan kelapa sawit menggunakan metode deep learning. Perancangan aplikasi ini dilatarbelakangi oleh 
    kebutuhan akan sistem pemantauan lahan kelapa sawit yang lebih efisien, akurat, dan berbasis teknologi. Metode konvensional 
    dalam pemantauan lahan seringkali memerlukan waktu dan biaya besar, serta kurang responsif terhadap perubahan kondisi di lapangan.
    Melalui pemanfaatan data citra satelit Sentinel-2 dan penerapan model semantic segmentation berbasis arsitektur U-Net dengan 
    encoder ResNet18 dan ResNet34, aplikasi ini dikembangkan untuk membantu proses analisis, visualisasi hasil segmentasi, dan 
    evaluasi kinerja model secara interaktif. Aplikasi ini memungkinkan pengguna untuk melihat hasil Confusion Matrix, metrik 
    evaluasi, serta perbandingan kinerja kedua model secara langsung, sehingga dapat memperlihatkan efektivitas penerapan metode 
    deep learning dalam pemantauan kebun kelapa sawit di Provinsi Riau.""")

# ---------- HALAMAN: TENTANG MODEL ----------
elif view == "model":
    st.markdown("<h2 style='text-align:center'>Tentang Model</h2>", unsafe_allow_html=True)
    st.divider()

    st.subheader("U-Net dengan ResNet18")
    st.info("""U-Net dengan ResNet18 merupakan model semantic segmentation yang menggabungkan arsitektur U-Net dengan encoder dari ResNet18.
    U-Net sendiri memiliki struktur encoder–decoder simetris: bagian encoder mengekstraksi fitur dari citra input, sedangkan decoder berfungsi mengembalikan resolusi spasial untuk menghasilkan peta segmentasi akhir.
    ResNet18 dipilih sebagai encoder karena terdiri dari 18 lapisan konvolusi dengan residual block, yang membantu mengatasi masalah vanishing gradient dan mempercepat konvergensi.
    Dengan kedalaman yang relatif rendah dibanding ResNet34, model ini memiliki lebih sedikit parameter dan waktu pelatihan yang lebih cepat, sehingga cocok untuk dataset dengan ukuran sedang atau keterbatasan sumber daya GPU.""")
    
    col_foto_curve_loss, col_foto_curve_accuracy = st.columns([1, 1], gap="large")
    with col_foto_curve_loss:
        st.image("resnet18_256x192_data/learning_curve_loss.png", width=500)
        
    with col_foto_curve_accuracy:
        st.image("resnet18_256x192_data/learning_curve_accuracy.png", width=500)
    
    col_foto_confusion_matrix, col_foto_metrik_evaluasi = st.columns([1, 1], gap="large")
    with col_foto_confusion_matrix:
        st.image("resnet18_256x192_data/confusion_matrix.png", width=500)
        
    with col_foto_metrik_evaluasi:
        st.image("resnet18_256x192_data/metrik_evaluasi.jpeg", width=500)



    st.subheader("U-Net dengan ResNet34")
    st.info("""U-Net dengan ResNet34 merupakan versi yang lebih kompleks dari arsitektur U-Net-ResNet18.
    Pada model ini, encoder menggunakan 34 lapisan konvolusi dengan jumlah residual block yang lebih banyak, memungkinkan model mengekstraksi fitur dengan representasi yang lebih mendalam dan detail.
    Kombinasi ini membuat model lebih mampu mengenali pola tekstur dan batas objek secara presisi pada citra, terutama pada data beresolusi tinggi seperti citra satelit.
    Namun, peningkatan kedalaman jaringan juga berarti lebih banyak parameter dan waktu pelatihan lebih lama, sehingga memerlukan sumber daya komputasi yang lebih besar.""")

    col_foto_curve_loss, col_foto_curve_accuracy = st.columns([1, 1], gap="large")
    with col_foto_curve_loss:
        st.image("resnet34_256x192_data/learning_curve_loss.png", width=500)
        
    with col_foto_curve_accuracy:
        st.image("resnet34_256x192_data/learning_curve_accuracy.png", width=500)

    col_foto_confusion_matrix, col_foto_metrik_evaluasi = st.columns([1, 1], gap="large")
    with col_foto_confusion_matrix:
        st.image("resnet34_256x192_data/confusion_matrix.png", width=500)
        
    with col_foto_metrik_evaluasi:
        st.image("resnet34_256x192_data/metrik_evaluasi.jpeg", width=500)

    st.header("Perbandingan Antar Model")
    st.info("""Nilai Wilks’ Lambda sebesar 0,7897 dan hasil uji statistik Fhitung = 2,8766 > Ftabel = 2,3861 menegaskan bahwa kinerja U-Net-ResNet34 berbeda secara signifikan dari U-Net-ResNet18. Dengan kedalaman jaringan yang lebih besar, ResNet34 mampu memberikan akurasi segmentasi lebih tinggi dan IoU yang lebih baik, terutama pada area kompleks atau vegetasi padat. 
            
Namun, peningkatan performa tersebut disertai dengan konsekuensi waktu pelatihan yang lebih panjang.
Model U-Net dengan ResNet18 memerlukan waktu training selama 4401,76 detik (±73,36 menit), sedangkan U-Net dengan ResNet34 membutuhkan waktu 4859,53 detik (±80,99 menit). Perbedaan ini menunjukkan bahwa kompleksitas arsitektur ResNet34 yang memiliki jumlah lapisan konvolusi lebih banyak menambah beban komputasi selama proses pelatihan. Meskipun demikian, waktu tambahan sekitar 7,6 menit tersebut menghasilkan peningkatan performa yang signifikan secara statistik. 
            
Secara keseluruhan, hasil uji F mendukung hipotesis bahwa ResNet34 memberikan peningkatan performa yang nyata dibanding ResNet18, dengan kompromi pada efisiensi waktu komputasi dalam konteks segmentasi wilayah perkebunan kelapa sawit.""")
