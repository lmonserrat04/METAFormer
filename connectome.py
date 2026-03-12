import numpy as np
import joblib
from nilearn.connectome import ConnectivityMeasure
from pathlib import Path
import argparse

# python3 connectome.py --path out_dir_aal/ABIDE_pcp/dparsf/filt_noglobal --output fc_aal
# python3 connectome.py --path out_dir_cc200/ABIDE_pcp/dparsf/filt_noglobal --output fc_cc200
# python3 connectome.py --path out_dir_dosenbach160/ABIDE_pcp/dparsf/filt_noglobal --output fc_dosenbach160

def generate_fc(path, kind='correlation', vectorize=True, discard_diagonal=True):
    """
    Generate functional connectivity matrix from 4D fMRI data
    :param arr: 4D fMRI data
    :param kind: type of functional connectivity matrix
    :return: functional connectivity matrix
    """
    arr = np.loadtxt(path)
    
    conn = ConnectivityMeasure(
        kind=kind, vectorize=vectorize, discard_diagonal=discard_diagonal)
    fc = conn.fit_transform([arr])[0]
    return fc


def main(args):
    path = Path(args.path).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    # Cambiamos glob por una lista directa para depurar
    files = list(path.iterdir())
    # Filtramos solo archivos que terminen en .1D (sensible a mayúsculas)
    files = [f for f in files if f.name.endswith('.1D')]

    print(f"Ruta procesada por Python: {path}")
    print(f"Archivos .1D encontrados: {len(files)}")

    for file in files:
        fc = generate_fc(file)
        np.savetxt(output.joinpath(file.name), fc, fmt='%.4f')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate functional connectivity matrix from atlas fMRI data')
    parser.add_argument('--path', type=str, help='path to fMRI data')
    parser.add_argument(
        '--output', type=str, help='path to output functional connectivity matrix directory')

    args = parser.parse_args()
    main(args)
