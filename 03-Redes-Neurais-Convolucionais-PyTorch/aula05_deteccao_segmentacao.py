# AULA 05 - DETECCAO E SEGMENTACAO - ALURA
# Modo rapido: python aula05_deteccao_segmentacao.py
# VOC real: python aula05_deteccao_segmentacao.py --voc --baixar

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / 'resultados'
DADOS = BASE / 'dados'

RESULTADOS.mkdir(exist_ok=True)
DADOS.mkdir(exist_ok=True)


def plotar_deteccao(imagem, objetos, nome):
    fig, ax = plt.subplots(figsize=(9, 6))

    # PyTorch: C,H,W -> Matplotlib: H,W,C
    ax.imshow(imagem.permute(1, 2, 0).numpy())

    for objeto in objetos:
        caixa = objeto['bndbox']

        xmin, ymin, xmax, ymax = [
            int(caixa[k])
            for k in ('xmin', 'ymin', 'xmax', 'ymax')
        ]

        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
        )

        ax.text(
            xmin,
            max(0, ymin - 8),
            objeto['name'],
            color='red',
            weight='bold'
        )

        print(
            'Objeto:', objeto['name'],
            '| caixa:', (xmin, ymin, xmax, ymax)
        )

    ax.set_title('Deteccao - classes e caixas delimitadoras')
    fig.tight_layout()
    fig.savefig(RESULTADOS / nome, dpi=140)
    plt.close(fig)

    print('Salvo:', RESULTADOS / nome)


def plotar_segmentacao(imagem, mascara, nome):
    rgb = imagem.permute(1, 2, 0).numpy()
    ids = mascara.squeeze(0).numpy()

    print('Imagem CHW:', tuple(imagem.shape))
    print('Mascara CHW:', tuple(mascara.shape))
    print('IDs de classes presentes:', np.unique(ids).tolist())

    # Pascal VOC:
    # 0 = fundo
    # 1 a 20 = classes de objetos
    # 255 = pixel ignorado
    ids_visuais = np.ma.masked_equal(ids, 255)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(rgb)
    axs[0].set_title('Imagem original')

    axs[1].imshow(
        ids_visuais,
        cmap='tab20',
        vmin=0,
        vmax=20,
        interpolation='nearest'
    )
    axs[1].set_title('Mascara: uma classe por pixel')

    axs[2].imshow(rgb)
    axs[2].imshow(
        ids_visuais,
        cmap='tab20',
        vmin=0,
        vmax=20,
        alpha=0.5,
        interpolation='nearest'
    )
    axs[2].set_title('Sobreposicao')

    for ax in axs:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(RESULTADOS / nome, dpi=140)
    plt.close(fig)

    print('Salvo:', RESULTADOS / nome)


def mostrar_formato_saidas(imagem):
    print('\nSAIDAS DIDATICAS DE CNNs - SEM TREINAMENTO')

    # N,C,H,W
    entrada = imagem.unsqueeze(0)

    # Deteccao simplificada:
    # regressao das 4 coordenadas de uma caixa.
    detector_simples = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4)
    )

    # Segmentacao:
    # uma pontuacao por classe em cada pixel.
    segmentador_simples = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 21, kernel_size=1)
    )

    with torch.no_grad():
        saida_caixa = detector_simples(entrada)
        saida_mascara = segmentador_simples(entrada)

    print('Entrada:', tuple(entrada.shape), '(N,C,H,W)')
    print(
        'Saida regressao:',
        tuple(saida_caixa.shape),
        '(4 coordenadas)'
    )
    print(
        'Saida segmentacao:',
        tuple(saida_mascara.shape),
        '(21 classes por pixel)'
    )

    print('OBS: esses modelos NAO foram treinados.')
    print(
        'Uma rede detectora completa tambem precisa '
        'prever classes e varios objetos.'
    )


def demonstracao():
    print('\nMODO DEMONSTRACAO - SEM DOWNLOAD')

    largura, altura = 420, 280

    # Criando imagem didatica com dois objetos.
    foto = Image.new(
        'RGB',
        (largura, altura),
        '#c1e6fa'
    )

    desenho = ImageDraw.Draw(foto)
    desenho.rectangle(
        (0, 232, largura, altura),
        fill='#82bb73'
    )

    # Mascara: cada pixel recebe um ID de classe.
    mascara_pil = Image.new(
        'L',
        (largura, altura),
        0
    )
    alvo = ImageDraw.Draw(mascara_pil)

    # Classe 1: pessoa.
    desenho.ellipse(
        (66, 36, 114, 84),
        fill='#f5bf91'
    )
    alvo.ellipse(
        (66, 36, 114, 84),
        fill=1
    )

    desenho.rectangle(
        (74, 84, 106, 170),
        fill='#ae4261'
    )
    alvo.rectangle(
        (74, 84, 106, 170),
        fill=1
    )

    for a, b in [
        ((80, 170), (70, 224)),
        ((102, 170), (111, 224))
    ]:
        desenho.line(
            (a, b),
            fill='#302e36',
            width=12
        )
        alvo.line(
            (a, b),
            fill=1,
            width=12
        )

    # Classe 2: carro.
    desenho.rectangle(
        (210, 160, 386, 210),
        fill='#f39a29'
    )
    alvo.rectangle(
        (210, 160, 386, 210),
        fill=2
    )

    desenho.polygon(
        [
            (240, 160),
            (259, 125),
            (344, 125),
            (367, 160)
        ],
        fill='#d77922'
    )
    alvo.polygon(
        [
            (240, 160),
            (259, 125),
            (344, 125),
            (367, 160)
        ],
        fill=2
    )

    for x in (247, 353):
        desenho.ellipse(
            (x - 17, 194, x + 17, 228),
            fill='#292929'
        )
        alvo.ellipse(
            (x - 17, 194, x + 17, 228),
            fill=2
        )

    # Estrutura de rotulos semelhante ao Pascal VOC.
    objetos = [
        {
            'name': 'person',
            'bndbox': {
                'xmin': 62,
                'ymin': 33,
                'xmax': 120,
                'ymax': 230
            }
        },
        {
            'name': 'car',
            'bndbox': {
                'xmin': 207,
                'ymin': 122,
                'xmax': 390,
                'ymax': 232
            }
        }
    ]

    rotulo = {
        'annotation': {
            'object': objetos
        }
    }

    # Transforma a imagem em Tensor.
    imagem = transforms.ToTensor()(foto)

    # Preserva os IDs das classes da mascara.
    mascara = transforms.PILToTensor()(mascara_pil)

    print('Imagem:', type(imagem), tuple(imagem.shape))
    print('Rotulo da deteccao:', type(rotulo), rotulo)
    print('Mascara:', type(mascara), tuple(mascara.shape))

    print(
        'Permuta:',
        tuple(imagem.permute(1, 2, 0).shape)
    )

    plotar_deteccao(
        imagem,
        rotulo['annotation']['object'],
        '09_deteccao_demo.png'
    )

    plotar_segmentacao(
        imagem,
        mascara,
        '10_segmentacao_demo.png'
    )

    mostrar_formato_saidas(imagem)


def pascal_voc(baixar):
    print('\nPASCAL VOC 2012 REAL')
    print('ATENCAO: o download pode ocupar varios GB.')

    # DATASET DE DETECCAO
    det = datasets.VOCDetection(
        root=str(DADOS),
        year='2012',
        image_set='val',
        download=baixar,
        transform=transforms.ToTensor()
    )

    imagem, rotulo = det[0]

    print('\nVOCDetection:')
    print('Imagem:', type(imagem), tuple(imagem.shape))
    print('Rotulo:', type(rotulo))
    print('Chaves:', list(rotulo.keys()))

    objetos = rotulo['annotation']['object']

    if isinstance(objetos, dict):
        objetos = [objetos]

    print('Quantidade de objetos:', len(objetos))

    plotar_deteccao(
        imagem,
        objetos,
        '11_voc_deteccao_real.png'
    )

    # DATASET DE SEGMENTACAO
    seg = datasets.VOCSegmentation(
        root=str(DADOS),
        year='2012',
        image_set='val',
        download=False,
        transform=transforms.ToTensor(),
        target_transform=transforms.PILToTensor()
    )

    foto, mascara = seg[0]

    print('\nVOCSegmentation:')
    print('Imagem:', type(foto), tuple(foto.shape))
    print('Mascara:', type(mascara), tuple(mascara.shape))

    plotar_segmentacao(
        foto,
        mascara,
        '12_voc_segmentacao_real.png'
    )

    mostrar_formato_saidas(imagem)

    print(
        'Observacao: indice 0 pode corresponder '
        'a imagens diferentes nos dois datasets.'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--voc',
        action='store_true',
        help='Usar Pascal VOC real'
    )

    parser.add_argument(
        '--baixar',
        action='store_true',
        help='Baixar Pascal VOC 2012'
    )

    args = parser.parse_args()

    print('=' * 58)
    print('AULA 05 - DETECCAO E SEGMENTACAO - ALURA')
    print('=' * 58)

    if args.voc:
        try:
            pascal_voc(args.baixar)
        except (RuntimeError, OSError, FileNotFoundError) as erro:
            print('ERRO ao acessar Pascal VOC:', erro)
            print(
                'Execute com --voc --baixar '
                'para obter o dataset.'
            )
            raise SystemExit(1)
    else:
        demonstracao()

    print('\nAULA 05 EXECUTADA COM SUCESSO!')
