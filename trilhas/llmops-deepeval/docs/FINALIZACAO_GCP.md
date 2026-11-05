# Finalizacao - GitHub Actions e Google Cloud

## Objetivo

Automatizar o deployment de uma aplicacao Docker
apos a aprovacao dos testes no GitHub Actions.

## Conteudos da aula

1. Google Cloud IAM
2. Contas de servico
3. Permissoes de deployment
4. GitHub Actions Secrets
5. Autenticacao no Google Cloud
6. Deployment Docker

## Metodo apresentado no curso

Criar uma conta de servico no Google Cloud.

Gerar uma chave JSON e cadastrar seu conteudo
como secret no repositorio GitHub.

Nome apresentado:
gcpsa-key

Nunca adicionar o JSON ao codigo ou ao Git.

## Metodo recomendado atualmente

Workload Identity Federation (OIDC).

Evita credenciais permanentes e permite
restringir acesso ao repositorio e branch.

## Situacao do projeto

Repositorio:
Rafaelaale/LLMOps-GitHubActions-DeepEval

Branch publicada:
Sanitization

Commit confirmado:
1cb312e

Workflow:
.github/workflows/evaluation.yml

Pendencias conhecidas:

- Corrigir a falha de validacao do GitHub Actions.
- Verificar as avaliacoes DeepEval.
- Preparar a branch main e a Pull Request.
- Configurar autenticacao GCP, se desejado.
- Definir destino de deployment e permissoes.
- Validar o pipeline antes de habilitar deploy.

O deployment permanece desativado.