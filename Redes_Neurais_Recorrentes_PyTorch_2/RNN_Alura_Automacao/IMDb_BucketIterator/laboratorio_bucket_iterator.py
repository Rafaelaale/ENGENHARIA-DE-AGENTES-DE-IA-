from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(42)
torch.set_num_threads(2)

origem = Path(sys.argv[1])
pasta = Path(__file__).resolve().parent

dados = torch.load(origem, map_location="cpu", weights_only=False)

X = dados["X"]
y = dados["y"]
comprimentos = dados["comprimentos"]
vocabulario = dados["vocabulario"]

PAD = vocabulario["<PAD>"]
BATCH_SIZE = 3

resultados = []

def testar(numero, descricao, condicao):
    assert condicao, f"FALHOU: {descricao}"
    resultados.append(f"{numero}. {descricao}: APROVADO")
    print(f"[APROVADO] {numero}. {descricao}")

print("=" * 68)
print("LABORATORIO - BUCKET ITERATOR")
print("=" * 68)

# Recuperar as sequencias sem padding.
sequencias = [
    X[:int(comprimentos[i]), i].clone()
    for i in range(X.shape[1])
]

class AvaliacoesDataset(Dataset):
    def __init__(self, sequencias, rotulos):
        self.sequencias = sequencias
        self.rotulos = rotulos

    def __len__(self):
        return len(self.sequencias)

    def __getitem__(self, indice):
        return (
            self.sequencias[indice],
            self.rotulos[indice]
        )

dataset = AvaliacoesDataset(sequencias, y)

testar(
    1,
    "Dataset recupera sequencias e rotulos",
    len(dataset) == len(y) == 6
)

# Agrupar indices por comprimento semelhante.
class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.indices = sorted(
            range(len(dataset)),
            key=lambda i: len(dataset.sequencias[i]),
            reverse=True
        )

    def __iter__(self):
        for inicio in range(0, len(self.indices), self.batch_size):
            yield self.indices[inicio:inicio + self.batch_size]

    def __len__(self):
        return (
            len(self.indices) + self.batch_size - 1
        ) // self.batch_size

sampler = BucketBatchSampler(dataset, BATCH_SIZE)

testar(
    2,
    "Bucket agrupa exemplos por comprimento",
    all(
        len(sequencias[sampler.indices[i]])
        >= len(sequencias[sampler.indices[i + 1]])
        for i in range(len(sampler.indices) - 1)
    )
)

# Criar batches no formato S x B.
def montar_batch(amostras):
    amostras = sorted(
        amostras,
        key=lambda item: len(item[0]),
        reverse=True
    )

    textos, rotulos = zip(*amostras)

    tamanhos = torch.tensor(
        [len(texto) for texto in textos],
        dtype=torch.long
    )

    batch = pad_sequence(
        textos,
        batch_first=False,
        padding_value=PAD
    )

    labels = torch.stack(rotulos).long()

    return batch, tamanhos, labels

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=montar_batch,
    num_workers=0
)

lotes = list(loader)

testar(
    3,
    "DataLoader cria batches automaticamente",
    len(lotes) == 2
)

for indice, (batch, tamanhos, rotulos) in enumerate(lotes, 1):
    print(f"\nBATCH {indice}")
    print("Shape:", tuple(batch.shape))
    print("Tamanhos:", tamanhos.tolist())
    print("Rotulos:", rotulos.tolist())

testar(
    4,
    "Batch organizado em S x B",
    all(batch.ndim == 2 for batch, _, _ in lotes)
)

testar(
    5,
    "Tamanhos ordenados do maior para o menor",
    all(
        tamanhos.tolist() == sorted(
            tamanhos.tolist(),
            reverse=True
        )
        for _, tamanhos, _ in lotes
    )
)

testar(
    6,
    "Padding completa sequencias menores",
    all(
        torch.all(batch[int(tamanhos[i]):, i] == PAD).item()
        for batch, tamanhos, _ in lotes
        for i in range(batch.shape[1])
    )
)

testar(
    7,
    "Rotulos mantem correspondencia com textos",
    all(
        batch.shape[1] == len(rotulos) == len(tamanhos)
        for batch, tamanhos, rotulos in lotes
    )
)

# Comparar padding global com padding por bucket.
padding_global = (
    X.shape[0] * X.shape[1]
    - int(comprimentos.sum().item())
)

padding_buckets = sum(
    batch.numel() - int(tamanhos.sum().item())
    for batch, tamanhos, _ in lotes
)

print("\nCOMPARACAO DE PADDING")
print("Padding global:", padding_global)
print("Padding por bucket:", padding_buckets)

testar(
    8,
    "Bucket nao aumenta padding neste conjunto",
    padding_buckets <= padding_global
)

# Embedding: indices S x B viram S x B x 100.
embedding = torch.nn.Embedding(
    num_embeddings=len(vocabulario),
    embedding_dim=100,
    padding_idx=PAD
)

with torch.no_grad():
    embedding.weight.copy_(dados["embedding_weights"])

primeiro_batch, primeiros_tamanhos, _ = lotes[0]
vetores = embedding(primeiro_batch)

testar(
    9,
    "Embedding recebe batch e retorna S x B x 100",
    vetores.shape == (
        primeiro_batch.shape[0],
        primeiro_batch.shape[1],
        100
    )
)

# Verificar dispositivo disponivel.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

batch_device = primeiro_batch.to(device)
tamanhos_cpu = primeiros_tamanhos.cpu()

testar(
    10,
    "Batch transferido ao dispositivo correto",
    batch_device.device.type == device.type
    and tamanhos_cpu.device.type == "cpu"
)

relatorio = pasta / "relatorio_bucket_iterator.txt"

relatorio.write_text(
    "AULA - BUCKET ITERATOR\n\n"
    + "\n".join(resultados)
    + "\n\n"
    + f"Amostras: {len(dataset)}\n"
    + f"Batch size: {BATCH_SIZE}\n"
    + f"Quantidade de batches: {len(lotes)}\n"
    + f"Padding global: {padding_global}\n"
    + f"Padding por bucket: {padding_buckets}\n"
    + f"Dispositivo: {device}\n"
    + "\n10/10 TESTES APROVADOS\n"
    + "Dados didaticos, nao IMDb oficial.\n"
    + "Sem treinamento nesta etapa.\n",
    encoding="utf-8"
)

print("\n" + "=" * 68)
print("RESULTADO FINAL: 10/10 TESTES APROVADOS")
print("Relatorio:", relatorio)