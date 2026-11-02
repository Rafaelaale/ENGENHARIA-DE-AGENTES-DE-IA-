import sys
import importlib

print("=" * 55)
print("AULA 03 - PREPARANDO O AMBIENTE")
print("=" * 55)

print("Python:", sys.version.split()[0])

bibliotecas = {
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "sklearn": "Scikit-learn",
    "skimage": "Scikit-image",
    "torch": "PyTorch",
    "torchvision": "Torchvision"
}

faltando = []

for modulo, nome in bibliotecas.items():
    try:
        biblioteca = importlib.import_module(modulo)
        print(f"{nome}: OK - {biblioteca.__version__}")
    except ImportError as erro:
        print(f"{nome}: NAO INSTALADO - {erro}")
        faltando.append(modulo)

if faltando:
    print("\nBibliotecas que precisam ser instaladas:")
    print(", ".join(faltando))
else:
    import torch

    print("\nCUDA disponivel:", torch.cuda.is_available())

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Dispositivo:", dispositivo)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    imagem = torch.rand(1, 3, 128, 128).to(dispositivo)

    camada = torch.nn.Conv2d(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        padding=1
    ).to(dispositivo)

    with torch.no_grad():
        resultado = camada(imagem)

    print("Imagem RGB:", tuple(imagem.shape))
    print("Saida CNN:", tuple(resultado.shape))

    assert resultado.shape == (1, 8, 128, 128)

    print("\nAMBIENTE PRONTO PARA AS PROXIMAS AULAS!")
