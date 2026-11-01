from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence
)

torch.manual_seed(42)
torch.set_num_threads(2)

pasta = Path(__file__).resolve().parent
origem = Path(sys.argv[1])

dados = torch.load(
    origem,
    map_location="cpu",
    weights_only=False
)

X_original = dados["X"]
comprimentos_originais = dados["comprimentos"]
vocabulario = dados["vocabulario"]
pesos_embedding = dados["embedding_weights"]

PAD = vocabulario["<PAD>"]
UNK = vocabulario["<UNK>"]

EMBED_DIM = 100
HIDDEN_SIZE = 256
BATCH_SIZE = 3

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

resultados = []

def testar(numero, descricao, condicao):
    assert condicao, f"TESTE {numero} FALHOU: {descricao}"
    resultados.append(
        f"{numero}. {descricao}: APROVADO"
    )
    print(f"[APROVADO] {numero}. {descricao}")

print("=" * 70)
print("LABORATORIO - SEQUENCIAS EMPACOTADAS")
print("=" * 70)
print("Dispositivo:", device)

# Recuperar sequencias originais.
sequencias = [
    X_original[:int(comprimentos_originais[i]), i].clone()
    for i in range(X_original.shape[1])
]

# Ordenar por comprimento, como sort_within_batch=True.
indices = sorted(
    range(len(sequencias)),
    key=lambda i: len(sequencias[i]),
    reverse=True
)

batches = []

for inicio in range(0, len(indices), BATCH_SIZE):
    grupo = indices[inicio:inicio + BATCH_SIZE]

    textos = [sequencias[i] for i in grupo]

    tamanhos = torch.tensor(
        [len(texto) for texto in textos],
        dtype=torch.long
    )

    X = pad_sequence(
        textos,
        batch_first=False,
        padding_value=PAD
    )

    rotulos = dados["y"][grupo]

    batches.append((X, tamanhos, rotulos))

testar(
    1,
    "Batches construidos com comprimentos variaveis",
    len(batches) == 2
)

testar(
    2,
    "Sequencias ordenadas do maior para o menor",
    all(
        tamanhos.tolist() == sorted(
            tamanhos.tolist(),
            reverse=True
        )
        for _, tamanhos, _ in batches
    )
)

class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        embed_vectors,
        ind_unk,
        ind_pad,
        hidden_size
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=ind_pad
        )

        with torch.no_grad():
            self.embedding.weight.copy_(embed_vectors)
            self.embedding.weight[ind_unk].zero_()
            self.embedding.weight[ind_pad].zero_()

        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False
        )

    def forward(self, X, tamanhos):
        embed = self.embedding(X)

        hidden_inicial = torch.zeros(
            1,
            X.size(1),
            self.hidden_size,
            device=X.device,
            dtype=embed.dtype
        )

        packed_input = pack_padded_sequence(
            embed,
            tamanhos.cpu(),
            batch_first=False,
            enforce_sorted=True
        )

        packed_output, hidden_final = self.rnn(
            packed_input,
            hidden_inicial
        )

        output, tamanhos_saida = pad_packed_sequence(
            packed_output,
            batch_first=False,
            total_length=X.size(0)
        )

        return {
            "embed": embed,
            "hidden_inicial": hidden_inicial,
            "packed_input": packed_input,
            "packed_output": packed_output,
            "output": output,
            "hidden_final": hidden_final,
            "tamanhos_saida": tamanhos_saida
        }

modelo = RNN(
    vocab_size=len(vocabulario),
    embed_dim=EMBED_DIM,
    embed_vectors=pesos_embedding,
    ind_unk=UNK,
    ind_pad=PAD,
    hidden_size=HIDDEN_SIZE
).to(device)

modelo.eval()

testar(
    3,
    "Modelo possui Embedding e GRU",
    isinstance(modelo.embedding, nn.Embedding)
    and isinstance(modelo.rnn, nn.GRU)
)

with torch.no_grad():
    for numero_batch, (X, tamanhos, rotulos) in enumerate(
        batches,
        start=1
    ):
        X = X.to(device)
        resultado = modelo(X, tamanhos)

        embed = resultado["embed"]
        packed_input = resultado["packed_input"]
        packed_output = resultado["packed_output"]
        output = resultado["output"]
        hidden_inicial = resultado["hidden_inicial"]
        hidden_final = resultado["hidden_final"]
        tamanhos_saida = resultado["tamanhos_saida"]

        S, B = X.shape

        print(f"\nBATCH {numero_batch}")
        print("X:", tuple(X.shape))
        print("embed:", tuple(embed.shape))
        print("hidden inicial:", tuple(hidden_inicial.shape))
        print(
            "packed_input.batch_sizes:",
            packed_input.batch_sizes.tolist()
        )
        print(
            "packed_output.batch_sizes:",
            packed_output.batch_sizes.tolist()
        )
        print("hidden final:", tuple(hidden_final.shape))
        print("output:", tuple(output.shape))
        print("tamanhos:", tamanhos.tolist())

        testar(
            4,
            f"Batch {numero_batch}: embedding S x B x 100",
            embed.shape == (S, B, EMBED_DIM)
        )

        testar(
            5,
            f"Batch {numero_batch}: hidden inicial 1 x B x 256",
            hidden_inicial.shape == (1, B, HIDDEN_SIZE)
        )

        testar(
            6,
            f"Batch {numero_batch}: empacotamento correto",
            packed_input.data.shape[0]
            == int(tamanhos.sum().item())
        )

        testar(
            7,
            f"Batch {numero_batch}: batch_sizes corretos",
            packed_input.batch_sizes.tolist()
            == [
                int((tamanhos > passo).sum().item())
                for passo in range(S)
            ]
        )

        testar(
            8,
            f"Batch {numero_batch}: packed_output preserva batch_sizes",
            torch.equal(
                packed_input.batch_sizes,
                packed_output.batch_sizes
            )
        )

        testar(
            9,
            f"Batch {numero_batch}: output S x B x 256",
            output.shape == (S, B, HIDDEN_SIZE)
        )

        testar(
            10,
            f"Batch {numero_batch}: comprimentos recuperados",
            torch.equal(
                tamanhos_saida.cpu(),
                tamanhos.cpu()
            )
        )

        # Para GRU unidirecional, hidden_final corresponde
        # a ultima saida valida de cada sequencia.
        ultimas_saidas = torch.stack([
            output[int(tamanhos[b]) - 1, b]
            for b in range(B)
        ])

        testar(
            11,
            f"Batch {numero_batch}: hidden ignora padding",
            torch.allclose(
                hidden_final[-1],
                ultimas_saidas,
                atol=1e-6
            )
        )

        # Saidas restauradas nas posicoes de padding
        # devem ser zeros.
        padding_zerado = all(
            torch.count_nonzero(
                output[int(tamanhos[b]):, b]
            ).item() == 0
            for b in range(B)
        )

        testar(
            12,
            f"Batch {numero_batch}: padding restaurado com zeros",
            padding_zerado
        )

relatorio = pasta / "relatorio_sequencias_empacotadas.txt"

relatorio.write_text(
    "AULA - SEQUENCIAS EMPACOTADAS\n\n"
    + "\n".join(resultados)
    + "\n\n"
    + f"Dispositivo: {device}\n"
    + f"Vocabulario: {len(vocabulario)}\n"
    + f"Embedding: {EMBED_DIM}\n"
    + f"Hidden: {HIDDEN_SIZE}\n"
    + f"Batches: {len(batches)}\n"
    + "\nTodos os testes aprovados.\n"
    + "Amostras didaticas, nao IMDb oficial.\n"
    + "Modelo nao treinado.\n",
    encoding="utf-8"
)

# Checkpoint novo, sem alterar modelos anteriores.
checkpoint = pasta / "modelo_gru_empacotamento_inicial.pt"

torch.save(
    {
        "model_state_dict": modelo.state_dict(),
        "vocabulario": vocabulario,
        "embed_dim": EMBED_DIM,
        "hidden_size": HIDDEN_SIZE,
        "treinado": False
    },
    checkpoint
)

print("\n" + "=" * 70)
print(f"RESULTADO FINAL: {len(resultados)}/{len(resultados)} TESTES APROVADOS")
print("Relatorio:", relatorio)
print("Checkpoint inicial:", checkpoint)