import pandas as pd
import os
from tqdm import tqdm
import wget
import requests
from pathlib import Path
from nilearn import datasets
import argparse
from concurrent.futures import ThreadPoolExecutor


log_path = Path(__file__).resolve().parent / "download_errors.txt"

#python3 download.py out_dir_cc200 --roi cc200
#python3 download.py out_dir_aal --roi aal
#python3 download.py out_dir_dos160 --roi dosenbach160
# base_str = https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/[strategy]/[derivative]/[file identifier]_[derivative].[ext]


# https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_aal/KKI_0050822_rois_aal.1D
base_str = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/[filt]/[roi]/[file identifier]_[roi].1D"


#roi_str = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/filt_global/[derivative]/[file identifier]_[derivative].1D"


def create_url(pipe, roi, fg):
    url = base_str.replace("[pipeline]", pipe)
    url = url.replace("[filt]", fg)
    url = url.replace("[roi]", roi)
    return url


def download_abide1_roi(pheno_file, out_dir, pipe, roi, fg):
    df = pd.read_csv(pheno_file)
    total = len(df)

    url = create_url(pipe, roi, fg)

    # create out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    problematic_subjects = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        sub_id = row["SUB_ID"]
        site = row["SITE_ID"]
        dx            = row['DX_GROUP']
        clasificacion = "ASD" if dx == 1 else "TC"
        filename = site + "_00" + str(sub_id)

        url = url.replace("[file identifier]",filename)
        out_file = f"{out_dir}/{site}_{sub_id}_{roi}.1D"
        prefix = f"[{i}/{total}]"
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))

                with open(out_file, "wb") as f, tqdm(
                    desc=f"{prefix} {site}/{clasificacion}/{filename}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            action = "sobreescrito" if dest_path.exists() else "descargado"
            print(f"✔ {prefix} {site}/{clasificacion}/{filename}")

        except Exception as e:
            print(f"❌ {prefix} Error {filename}: {e}")
            problematic_subjects.append((filename, site, clasificacion, str(e)))

    # ─── LOG DE ERRORES ───────────────────────────────────────────────────────────
    if problematic_subjects:
        with open(log_path, "w") as log:
            log.write(f"Sujetos con error de descarga ({len(problematic_subjects)}):\n\n")
            for file_id, site, clasificacion, error in problematic_subjects:
                log.write(f"  {site}/{clasificacion}/{file_id} → {error}\n")
        print(f"\n⚠ {len(problematic_subjects)} errores. Log guardado en: {log_path}")
    else:
        print("\n✅ Todos los sujetos descargados y ordenados correctamente.")


        

def download_abide1_pcp(pheno_file, out_dir):
    df = pd.read_csv(pheno_file)

    # create out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        sub_id = row["SUB_ID"]
        site = row["SITE_ID"]
        url = base_str.replace("[file identifier]", site + "_00" + str(sub_id))
        out_file = f"{out_dir}/{site}_{sub_id}_func_preproc.nii.gz"
        try:
            wget.download(url, out_file)
        except Exception as e:
            print(e)
            print(f"Failed to download {url} to {out_file}")


def download_single_atlas(pipe, roi, use_fg):
    """
    Crea una carpeta local y descarga el atlas específico.
    """
    # Definición dinámica del directorio de salida
    target_dir = os.path.join(os.getcwd(), f"out_dir_{roi}")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"🚀 [HILO] Iniciando: Atlas={roi} | Carpeta={target_dir}")
    
    try:
        # Nilearn descargará los datos dentro de target_dir/ABIDE_pcp/...
        datasets.fetch_abide_pcp(
            data_dir=target_dir,
            pipeline=pipe,
            derivatives=[f'rois_{roi}'],
            band_pass_filtering=use_fg,
            global_signal_regression=False
        )
        return f"✅ {roi}: Descargado en {target_dir}"
    except Exception as e:
        return f"❌ {roi}: Error -> {str(e)[:100]}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga ABIDE con carpetas automáticas")
    parser.add_argument("--pipe", type=str, default="dparsf", help="Pipeline (dparsf/cpac)")
    parser.add_argument("--rois", nargs="+", default=["cc200", "dosenbach160", "aal"], help="Atlas a descargar")
    
    # Lógica de booleano por defecto True
    parser.add_argument("--fg", action="store_true", default=True, help="Filtrado activo (Default)")
    parser.add_argument("--no-fg", action="store_false", dest="fg", help="Desactivar filtrado")

    args = parser.parse_args()

    print(f"📂 Ejecutando en: {os.getcwd()}")
    print(f"🌐 Atlas seleccionados: {args.rois}")
    print("-" * 40)

    # Paralelización
    with ThreadPoolExecutor(max_workers=len(args.rois)) as executor:
        futures = [
            executor.submit(download_single_atlas, args.pipe, roi, args.fg)
            for roi in args.rois
        ]
        
        results = [f.result() for f in futures]

    print("\n--- Estado Final ---")
    for res in results:
        print(res)