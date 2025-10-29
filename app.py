# app.py
import os, io, tempfile, zipfile, warnings, hashlib
import numpy as np
import streamlit as st
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
def scale_per_band_valid(arr, nodata, low=2, high=98):
    # Mendefinisikan fungsi untuk menormalkan nilai citra per-band ke rentang [0, 1] memakai persentil.
    x = arr.astype("float32")
    # konversi array input ke float32 untuk presisi
    if x.ndim == 2: x = x[..., None]
    # Jika input 2D (H×W), tambahkan dimensi channel di belakang ⇒ (H×W×1).
    if nodata is not None:
        #Jika ada nilai NoData yang didefinisikan
        for c in range(x.shape[-1]):
        # Loop tiap channel c.
            band = x[..., c]
            # Ambil view band ke-c (band adalah view ke x[..., c]).
            band[band == nodata] = np.nan
            # Ganti semua piksel yang persis sama dengan nodata menjadi NaN.
    out = np.empty_like(x)
    # Inisialisasi array output dengan bentuk dan tipe yang sama seperti x.
    for c in range(x.shape[-1]):
    # Loop tiap channel c.
        band = x[..., c]
        # Ambil view band ke-c (band adalah view ke x[..., c]).
        lo = np.nanpercentile(band, low)
        hi = np.nanpercentile(band, high)
        # Hitung persentil bawah/atas (mis. 2% & 98%) dengan mengabaikan NaN
        out[..., c] = np.clip((band-lo)/(hi-lo+1e-6), 0, 1)
        # band - lo = menggeser nilai citra supaya batas bawah (lo) menjadi nol.
        # (hi - lo + 1e-6) = skala rentang nilai citra supaya batas atas (hi) menjadi 1.
    return np.nan_to_num(out, nan=0.0)
    # Mengembalikan array float32 bernilai di [0, 1] per band, tanpa NaN.

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
    """
    Ambil luas piksel (m²).
    Jika PIXEL_AREA_M2 != None → gunakan nilai tersebut.
    Jika None → baca dari GeoTransform (|a*e|). Jika tidak valid → fallback.
    """

    with rasterio.open(tif_path) as ds:
        a = float(ds.transform.a)
        e = float(ds.transform.e)
        area = abs(a * e)
        if not np.isfinite(area) or area <= 0:
            area = float(PIXEL_AREA_M2)
        return area

def count_pixels_eq1(tif_path: str) -> int:
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)
        return int((arr == 1).sum())

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
def predict_full_raster(raster_path, model, in_ch=3, tile=256, overlap=64, multiple=32, disp_bands=(1,2,3)):
    assert tile % multiple == 0 and 0 <= overlap < tile
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

                patch01 = scale_per_band_valid(patch, nodata, 2, 98)
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
def save_mask_geotiff_like(ref_path, mask01, out_path, nodata=0, compress="lzw"):
    # ref_path: path raster referensi (citra input) untuk mengambil CRS & transform.
    # compress: skema kompresi GeoTIFF (default "lzw").
    if mask01.ndim == 3: mask01 = mask01[..., 0]
    # Jika mask berdimensi 3 (H,W,1), ambil channel pertama → jadi 2D (H,W). 
    # Penyederhanaan karena GeoTIFF yang ditulis hanya 1 band.
    mask01 = mask01.astype(np.uint8)

    with rasterio.open(ref_path) as src:
        src_prof = src.profile.copy()
        # salin profil (metadata) dari raster referensi
        crs = src_prof.get("crs", None)
        # ambil CRS dari profil (bisa None, mis. UTM EPSG:xxxxx)
        transform = src_prof.get("transform", src.transform)
        # ambil transform dari profil (jika tidak ada, pakai src.transform langsung)
        height, width = mask01.shape
        # ambil tinggi dan lebar dari mask01

    prof = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1, # hanya 1 band
        "dtype": rasterio.uint8,
        "crs": crs, 
        "transform": transform, 
        "compress": compress, # kompresi LZW
        "BIGTIFF": "IF_SAFER", # aktifkan BigTIFF otomatis jika ukuran > 4GB.
        "tiled": True, # penyimpanan bertipe tile
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "pixel",
        "photometric": "MINISBLACK", # 0 dianggap gelap, nilai lebih tinggi lebih terang
        "nodata": nodata,
    }

    with rasterio.open(out_path, "w", **prof) as dst:
    # Python akan melakukan “argument unpacking”, artinya:
    # dictionary prof dipecah menjadi serangkaian argumen keyword.
        dst.write(mask01, 1)
        # dst.write(mask01, 1) → tulis array ke band 1.
        try:
            dst.write_colormap(1, {0: (0, 0, 0, 0), 1: (255, 248, 113, 255)})
            # dst.write_colormap(1, {...}) → pasang colormap untuk band 1:
        except Exception:
            pass

    if BUILD_OVERVIEWS:
        with rasterio.open(out_path, "r+") as dst:
            # Mode "r+" → buka untuk read/write metadata tambahan.
            dst.build_overviews([2, 4, 8, 16, 32], resampling=Resampling.nearest)
            # build_overviews([2,4,8,16,32], ...) → buat pyramid pada faktor downsample 2×, 4×, 8×, 16×, 32×.
            # Resampling.nearest → tepat untuk mask (kategorikal); menghindari interpolasi nilai
            dst.update_tags(ns="rio_overview", resampling="nearest")

# ---- Visual PNG (citra asli & overlay) ----
def _vis_global_gamma(a, lows, highs, gamma=1.5):
    vis01 = scale_fixed_minmax(a, lows, highs)
    if vis01.shape[-1] == 1:
        vis01 = np.repeat(vis01, 3, axis=-1)
    return apply_gamma(vis01, gamma=gamma)

def save_png_preview_from_tif(tif_path, out_png, lows, highs, max_side=2000, is_mask=False):
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        scale = max(H/max_side, W/max_side, 1.0)
        dH, dW = int(H/scale), int(W/scale)

        if is_mask:
            a = src.read(1, out_shape=(dH, dW), resampling=Resampling.nearest).astype(np.uint8)
            img = Image.fromarray(a, mode="P")
            palette = [0,0,0, 255,248,113] + [0,0,0]*254
            img.putpalette(palette)
            img.save(out_png, optimize=True, transparency=0)
        else:
            idxs = [1,2,3] if src.count >= 3 else [1]
            a = src.read(indexes=idxs, out_shape=(len(idxs), dH, dW),
                         resampling=Resampling.bilinear).transpose(1,2,0)
            vis01 = _vis_global_gamma(a, lows, highs, gamma=1.5)
            Image.fromarray((np.clip(vis01,0,1)*255).astype(np.uint8), mode="RGB").save(out_png, optimize=True)

def save_overlay_png_from_tif(base_tif, mask_tif, out_png, lows, highs,
                              max_side=2000, color_fg=(0, 255, 0), color_bg=(255, 0, 0), alpha=0.65):
    with rasterio.open(base_tif) as src_b:
        Hb, Wb = src_b.height, src_b.width
        scale = max(Hb/max_side, Wb/max_side, 1.0)
        dH, dW = int(Hb/scale), int(Wb/scale)

        if src_b.count >= 3:
            idxs = [1,2,3]
            base_raw = src_b.read(indexes=idxs, out_shape=(len(idxs), dH, dW),
                                  resampling=Resampling.bilinear).transpose(1,2,0)
        else:
            base1 = src_b.read(1, out_shape=(dH, dW), resampling=Resampling.bilinear)
            base_raw = np.repeat(base1[..., None], 3, axis=-1)

        if src_b.nodata is not None:
            band1_small = src_b.read(1, out_shape=(dH, dW), resampling=Resampling.nearest)
            valid = (band1_small != src_b.nodata)
        else:
            valid = np.any(base_raw != 0, axis=-1)

        base01 = _vis_global_gamma(base_raw, lows, highs, gamma=1.5)
        over   = (np.clip(base01, 0, 1) * 255.0).astype(np.float32)

    with rasterio.open(mask_tif) as src_m:
        mask_small = src_m.read(1, out_shape=(dH, dW), resampling=Resampling.nearest).astype(np.uint8)

    rf,gf,bf = color_fg
    rb,gb,bb = color_bg
    a = float(alpha)

    m1 = (mask_small == 1) & valid
    if m1.any():
        over[m1, 0] = (1 - a) * over[m1, 0] + a * rf
        over[m1, 1] = (1 - a) * over[m1, 1] + a * gf
        over[m1, 2] = (1 - a) * over[m1, 2] + a * bf

    m0 = (mask_small == 0) & valid
    if m0.any():
        over[m0, 0] = (1 - a) * over[m0, 0] + a * rb
        over[m0, 1] = (1 - a) * over[m0, 1] + a * gb
        over[m0, 2] = (1 - a) * over[m0, 2] + a * bb

    Image.fromarray(np.clip(over, 0, 255).astype(np.uint8), mode="RGB").save(out_png, optimize=True)

# ---- Polygonize mask menjadi Shapefile ----
def mask_tif_to_shapefile(mask_tif_path, shp_out_path, only_value=1, dissolve=False):
    with rasterio.open(mask_tif_path) as ds:
        arr = ds.read(1)
        tr  = ds.transform
        crs = ds.crs

    polys = [shp_shape(geom) for geom, val in shapes(arr, mask=(arr == only_value), transform=tr)
             if int(val) == int(only_value)]

    gdf = gpd.GeoDataFrame({"value":[only_value]*len(polys)}, geometry=polys, crs=crs)
    if gdf.empty:
        gdf = gpd.GeoDataFrame({"value":[], "area_m2":[]}, geometry=[], crs=crs)
        gdf.to_file(shp_out_path, driver="ESRI Shapefile")
        return shp_out_path

    gdf["area_m2"] = gdf.geometry.area.astype(float)
    if dissolve:
        gdf = gdf.dissolve(by="value", as_index=False, aggfunc={"area_m2":"sum"})
    gdf.to_file(shp_out_path, driver="ESRI Shapefile")
    return shp_out_path

# ---- ZIP: mask TIF + shapefile + overlay + input asli ----
def zip_mask_and_shapefile(mask_tif_path: str, shp_path: str, overlay_png: str = None, input_original: str = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.isfile(mask_tif_path):
            zf.write(mask_tif_path, arcname=os.path.basename(mask_tif_path))
        stem, _ = os.path.splitext(shp_path)
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix", ".fix", ".xml"]:
            p = stem + ext
            if os.path.isfile(p):
                zf.write(p, arcname=os.path.basename(p))
        if overlay_png and os.path.isfile(overlay_png):
            zf.write(overlay_png, arcname=os.path.basename(overlay_png))
        if input_original and os.path.isfile(input_original):
            zf.write(input_original, arcname=os.path.basename(input_original))
    buf.seek(0)
    return buf.getvalue()







# ====================== UI / ROUTING ======================
st.set_page_config(page_title="Deteksi Sawit – U-Net ResNet", layout="wide")

# gunakan query param untuk "routing" internal
qp = st.query_params
view = qp.get("view", "home")

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
        tile=TILE, overlap=OVERLAP, multiple=MULT, disp_bands=DISPLAY_BANDS
    )
    mask = (prob[...,0] >= THRESH).astype(np.uint8)
    # Mengubah probabilitas jadi mask biner:

    mask_tif_path = os.path.join(work_dir, "prediction_mask.tif")
    # Path output untuk GeoTIFF mask.
    save_mask_geotiff_like(in_path, mask, mask_tif_path, nodata=0)

    # Hitung luas area terdeteksi (mask==1)
    px_area_m2   = pixel_area_m2_from_tif(mask_tif_path, PIXEL_AREA_M2)
    n_pixels_pos = count_pixels_eq1(mask_tif_path)
    area_m2      = n_pixels_pos * px_area_m2
    area_ha      = area_m2 / 10_000.0
    area_km2     = area_m2 / 1_000_000.0

    shp_path = os.path.join(work_dir, "prediction_mask.shp")
    mask_tif_to_shapefile(mask_tif_path, shp_path, only_value=1, dissolve=True)

    png_rgb_path = os.path.join(work_dir, "input_preview_rgb.png")
    png_ovr_path = os.path.join(work_dir, "input_overlay.png")
    save_png_preview_from_tif(in_path, png_rgb_path, GLOBAL_LOW, GLOBAL_HIGH, max_side=PNG_MAX_SIDE, is_mask=False)
    save_overlay_png_from_tif(in_path, mask_tif_path, png_ovr_path, GLOBAL_LOW, GLOBAL_HIGH, max_side=PNG_MAX_SIDE)

    with open(png_rgb_path, "rb") as f: png_rgb_bytes = f.read()
    with open(png_ovr_path, "rb") as f: png_ovr_bytes = f.read()
    with open(mask_tif_path, "rb") as f: mask_tif_bytes = f.read()

    zip_bytes = zip_mask_and_shapefile(mask_tif_path, shp_path,
                                       overlay_png=png_ovr_path, input_original=in_path)

    return {
        "work_dir": work_dir,
        "png_rgb_bytes": png_rgb_bytes,
        "png_ovr_bytes": png_ovr_bytes,
        "mask_tif_bytes": mask_tif_bytes,
        "zip_bytes": zip_bytes,
        "zip_name": f"{os.path.splitext(filename)[0]}_results.zip",
        "mask_name": "prediction_mask.tif",
        # (BARU) metrik luas
        "n_pixels_pos": n_pixels_pos,
        "pixel_area_m2": px_area_m2,
        "area_m2": area_m2,
        "area_ha": area_ha,
        "area_km2": area_km2,
        "fmt_area": fmt_area_triple(area_m2),
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
        st.session_state.pop("result", None)
    if c2.button("ResNet34", type="primary" if st.session_state.model_sel.endswith("res34_backbone_100epochs_256x192_compiled.hdf5") else "secondary"):
        st.session_state.model_sel = "models/res34_backbone_100epochs_256x192_compiled.hdf5"
        st.session_state.pop("result", None)

    st.subheader("Upload citra (TIF/JP2)")
    uploaded = st.file_uploader(" ", type=["tif","tiff","jp2"], accept_multiple_files=False, label_visibility="collapsed")
    if uploaded is not None:
        new_key = hashlib.md5(uploaded.getbuffer()).hexdigest() + "|" + st.session_state.model_sel
        if st.session_state.get("last_key") != new_key:
            st.session_state.pop("result", None)
            st.session_state["last_key"] = new_key

    process = st.button("Proses", use_container_width=True, type="primary")

    if process:
        if uploaded is None:
            st.error("Silakan upload 1 file citra terlebih dahulu.")
            st.stop()
        with st.spinner("Inferensi berjalan..."):
            # manggil inference pipeline
            st.session_state["result"] = run_pipeline(
                uploaded.getbuffer(), uploaded.name, st.session_state.model_sel
            )
        st.success("Selesai diproses.")
        st.query_params["view"] = "hasil"   # tampilkan halaman hasil TANPA menu sidebar
        st.rerun()

# ---------- HALAMAN: HASIL  ----------
elif view == "hasil":
    st.markdown("<h2 style='text-align:center'>Hasil Pengujian</h2>", unsafe_allow_html=True)
    st.divider()

    res = st.session_state.get("result")
    if res is None:
        st.warning("Belum ada hasil. Buka menu Pengujian untuk memproses citra.")
        if st.button("⬅️ Ke Pengujian", use_container_width=True):
            st.query_params["view"] = "pengujian"
            st.rerun()
        st.stop()

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

    # (BARU) tampilkan metrik luas
    st.divider()
    st.subheader("Luas Area Kebun Kelapa Sawit Terdeteksi")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Jumlah piksel positif", f"{res['n_pixels_pos']:,}")
    with c2:
        st.metric("Luas/piksel", f"{res['pixel_area_m2']:.2f} m²")
    with c3:
        st.metric("Luas total (ha)", f"{res['area_ha']:.2f} ha")
    with c4:
        st.metric("Luas total (km²)", f"{res['area_km2']:.4f} km²")

    st.caption(f"≈ {res['area_m2']:,.2f} m² total (dihitung dari {res['n_pixels_pos']:,} piksel × {res['pixel_area_m2']:.2f} m²/piksel).")

    st.divider()
    st.download_button(
        "Unduh hasil (mask tif + shapefile + overlay PNG + input asli)",
        data=res["zip_bytes"],
        file_name=res["zip_name"],
        mime="application/zip",
        use_container_width=True
    )
    if st.button("Kembali ke Pengujian", use_container_width=True):
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
