"""
Projeto RMS Data Hub
====================
Aplicação FastAPI + SQLAlchemy, em arquivo único, para extração,
armazenamento e consulta de dados dos municípios da Região Metropolitana
de Sorocaba (RMS) e de dados de mercado de trabalho (profissões, salários e
estabelecimentos) da RAIS/base dos dados para a região.

Como executar:
    pip install fastapi uvicorn sqlalchemy httpx
    python app.py

Depois acesse: http://127.0.0.1:8000

Dados externos (opcionais):
    Coloque os 4 arquivos CSV originais na pasta "dados/", ao lado deste
    app.py, com os seguintes nomes (o app detecta e importa automaticamente
    na primeira inicialização, se as tabelas estiverem vazias):

        dados/profissoes_clt.csv        -> trabalhadores CLT por profissão/município
        dados/profissoes_nao_clt.csv    -> trabalhadores não-CLT por profissão/município
        dados/profissoes_total.csv      -> total de trabalhadores por profissão/município
        dados/estabelecimentos_rais.csv -> microdados de estabelecimentos (RAIS 2021)

    Se a pasta "dados/" ou algum desses arquivos não existir, o app funciona
    normalmente — apenas a seção de "Mercado de Trabalho" fica vazia.
"""

import asyncio
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DO BANCO DE DADOS (SQLAlchemy)
# ---------------------------------------------------------------------------
# Padrão: SQLite local para testes instantâneos. Para trocar de banco, basta
# alterar a variável DATABASE_URL abaixo (ex.: PostgreSQL ou MySQL).
#
#   PostgreSQL: "postgresql+psycopg2://usuario:senha@host:5432/nome_banco"
#   MySQL:      "mysql+pymysql://usuario:senha@host:3306/nome_banco"
#
# Em produção (ex.: Render), defina a variável de ambiente DATABASE_URL para
# apontar a um banco gerenciado (ex.: PostgreSQL) — assim os dados não se
# perdem a cada novo deploy, já que o disco local do serviço é temporário.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./rms_dados.db")

# connect_args só é necessário para SQLite (permite uso em múltiplas threads,
# como as usadas pelo Uvicorn/FastAPI). Para PostgreSQL/MySQL, remova esse
# parâmetro ou deixe um dicionário vazio.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Pasta com os CSVs externos (opcionais) a serem importados na inicialização.
# Ver docstring no topo do arquivo para os nomes esperados.
DADOS_DIR = Path(__file__).resolve().parent / "dados"


def get_db():
    """Dependency do FastAPI: fornece uma sessão de banco por requisição."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 2. MODELAGEM DA TABELA (ORM)
# ---------------------------------------------------------------------------
class Municipio(Base):
    __tablename__ = "municipios"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nome = Column(String(120), nullable=False, unique=True, index=True)
    estado = Column(String(2), nullable=False, default="SP")
    populacao_estimada = Column(Integer, nullable=True)
    pib = Column(Float, nullable=True)  # PIB a preços correntes (mil reais)
    data_atualizacao = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class Profissao(Base):
    """
    Dados de trabalhadores por profissão e município, vindos de 3 arquivos
    complementares (identificados pelo campo `tipo_vinculo`):

      - "CLT"      -> trabalhadores registrados em regime CLT
      - "NAO_CLT"  -> trabalhadores em outros regimes (estatutário, autônomo etc.)
      - "TOTAL"    -> total geral de trabalhadores na profissão/município
    """

    __tablename__ = "profissoes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    nome_municipio = Column(String(120), nullable=False, index=True)
    nome_profissao = Column(String(200), nullable=False, index=True)
    tipo_vinculo = Column(String(10), nullable=False, index=True)
    total_trabalhadores = Column(Integer, nullable=True)
    salario_medio = Column(Float, nullable=True)


class Estabelecimento(Base):
    """
    Microdados de estabelecimentos (RAIS 2021, via basedosdados.org) para os
    municípios da região de Sorocaba. Cada linha representa um
    estabelecimento (não um município), permitindo agregações por região,
    porte, natureza jurídica etc.
    """

    __tablename__ = "estabelecimentos"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ano = Column(Integer, nullable=True, index=True)
    id_municipio = Column(String(10), nullable=True)
    nome_municipio = Column(String(120), nullable=False, index=True)
    quantidade_vinculos_ativos = Column(Integer, nullable=True)
    quantidade_vinculos_clt = Column(Integer, nullable=True)
    quantidade_vinculos_estatutarios = Column(Integer, nullable=True)
    natureza_juridica = Column(String(10), nullable=True)
    tamanho_estabelecimento = Column(String(10), nullable=True)
    tipo_estabelecimento = Column(String(10), nullable=True)
    indicador_simples = Column(String(5), nullable=True)
    cnae_2_subclasse = Column(String(20), nullable=True)
    subsetor_ibge = Column(String(10), nullable=True)
    cep = Column(String(10), nullable=True)


# ---------------------------------------------------------------------------
# 3. SCHEMAS (Pydantic) - saída da API
# ---------------------------------------------------------------------------
class MunicipioSchema(BaseModel):
    id: int
    nome: str
    estado: str
    populacao_estimada: Optional[int] = None
    pib: Optional[float] = None
    data_atualizacao: datetime

    class Config:
        from_attributes = True


class ProfissaoSchema(BaseModel):
    id: int
    nome_municipio: str
    nome_profissao: str
    tipo_vinculo: str
    total_trabalhadores: Optional[int] = None
    salario_medio: Optional[float] = None

    class Config:
        from_attributes = True


class EstabelecimentoResumoSchema(BaseModel):
    nome_municipio: str
    total_estabelecimentos: int
    total_vinculos_ativos: int
    total_vinculos_clt: int
    total_vinculos_estatutarios: int


# ---------------------------------------------------------------------------
# 4. DADOS OFICIAIS DA RMS (27 municípios)
# ---------------------------------------------------------------------------
# Região Metropolitana de Sorocaba, instituída pela Lei Complementar Estadual
# nº 1.241/2014 (SP), composta por 27 municípios. Os valores de população e
# PIB abaixo são valores de referência (podem ser sobrescritos pelo endpoint
# de sincronização com o IBGE em /api/ibge/sincronizar).
MUNICIPIOS_RMS_SEED = [
    {"nome": "Sorocaba", "populacao_estimada": 695300, "pib": 27500000.0},
    {"nome": "Votorantim", "populacao_estimada": 121600, "pib": 2650000.0},
    {"nome": "Itu", "populacao_estimada": 175600, "pib": 6800000.0},
    {"nome": "Salto", "populacao_estimada": 122100, "pib": 3300000.0},
    {"nome": "Salto de Pirapora", "populacao_estimada": 41500, "pib": 850000.0},
    {"nome": "Mairinque", "populacao_estimada": 47700, "pib": 1050000.0},
    {"nome": "São Roque", "populacao_estimada": 96700, "pib": 1900000.0},
    {"nome": "Ibiúna", "populacao_estimada": 84300, "pib": 1450000.0},
    {"nome": "Piedade", "populacao_estimada": 53200, "pib": 850000.0},
    {"nome": "Tatuí", "populacao_estimada": 122400, "pib": 2600000.0},
    {"nome": "Boituva", "populacao_estimada": 61200, "pib": 1750000.0},
    {"nome": "Cerquilho", "populacao_estimada": 53100, "pib": 1300000.0},
    {"nome": "Cesário Lange", "populacao_estimada": 18300, "pib": 380000.0},
    {"nome": "Porto Feliz", "populacao_estimada": 57400, "pib": 1550000.0},
    {"nome": "Cabreúva", "populacao_estimada": 51800, "pib": 3200000.0},
    {"nome": "Araçariguama", "populacao_estimada": 27300, "pib": 950000.0},
    {"nome": "Alumínio", "populacao_estimada": 22200, "pib": 620000.0},
    {"nome": "Araçoiaba da Serra", "populacao_estimada": 37600, "pib": 720000.0},
    {"nome": "Capela do Alto", "populacao_estimada": 22400, "pib": 380000.0},
    {"nome": "Iperó", "populacao_estimada": 34600, "pib": 780000.0},
    {"nome": "Pilar do Sul", "populacao_estimada": 39500, "pib": 650000.0},
    {"nome": "São Miguel Arcanjo", "populacao_estimada": 34800, "pib": 620000.0},
    {"nome": "Tapiraí", "populacao_estimada": 9200, "pib": 180000.0},
    {"nome": "Vargem Grande Paulista", "populacao_estimada": 51900, "pib": 900000.0},
    {"nome": "Alambari", "populacao_estimada": 6300, "pib": 150000.0},
    {"nome": "Jumirim", "populacao_estimada": 6200, "pib": 140000.0},
    {"nome": "Sarapuí", "populacao_estimada": 16400, "pib": 280000.0},
]

assert len(MUNICIPIOS_RMS_SEED) == 27, "A RMS deve conter 27 municípios."


def seed_database():
    """Popula o banco na primeira execução, caso a tabela esteja vazia."""
    db = SessionLocal()
    try:
        total = db.query(func.count(Municipio.id)).scalar()
        if total and total > 0:
            return  # já populado, não faz nada

        agora = datetime.now(timezone.utc)
        for item in MUNICIPIOS_RMS_SEED:
            municipio = Municipio(
                nome=item["nome"],
                estado="SP",
                populacao_estimada=item["populacao_estimada"],
                pib=item["pib"],
                data_atualizacao=agora,
            )
            db.add(municipio)
        db.commit()
    finally:
        db.close()


def _ler_float(valor: str) -> Optional[float]:
    """Converte string de CSV em float, tratando valores vazios/ausentes."""
    if valor is None:
        return None
    valor = valor.strip()
    if not valor:
        return None
    try:
        return float(valor)
    except ValueError:
        return None


def _ler_int(valor: str) -> Optional[int]:
    """Converte string de CSV em int (via float, para tolerar '123.0')."""
    numero = _ler_float(valor)
    return int(numero) if numero is not None else None


def _importar_profissoes_csv(db: Session, caminho: Path, coluna_total: str, tipo_vinculo: str) -> int:
    """
    Lê um CSV de profissões (colunas: nome_municipio, [setor_cnae,]
    nome_profissao, <coluna_total>, salario_medio) e insere os registros na
    tabela `profissoes`, marcados com o `tipo_vinculo` informado.
    Retorna a quantidade de linhas importadas.
    """
    if not caminho.is_file():
        return 0

    registros = []
    with caminho.open(encoding="utf-8", newline="") as arquivo_csv:
        leitor = csv.DictReader(arquivo_csv)
        for linha in leitor:
            nome_municipio = (linha.get("nome_municipio") or "").strip()
            nome_profissao = (linha.get("nome_profissao") or "").strip()
            if not nome_municipio or not nome_profissao:
                continue
            registros.append(
                {
                    "nome_municipio": nome_municipio,
                    "nome_profissao": nome_profissao,
                    "tipo_vinculo": tipo_vinculo,
                    "total_trabalhadores": _ler_int(linha.get(coluna_total)),
                    "salario_medio": _ler_float(linha.get("salario_medio")),
                }
            )

    if registros:
        db.bulk_insert_mappings(Profissao, registros)
        db.commit()
    return len(registros)


def _importar_estabelecimentos_csv(db: Session, caminho: Path) -> int:
    """
    Lê o CSV de microdados de estabelecimentos (RAIS) e insere os registros
    na tabela `estabelecimentos`, em lotes, para não sobrecarregar a memória
    em arquivos grandes. Retorna a quantidade de linhas importadas.
    """
    if not caminho.is_file():
        return 0

    TAMANHO_LOTE = 5000
    total_importado = 0
    lote = []

    with caminho.open(encoding="utf-8", newline="") as arquivo_csv:
        leitor = csv.DictReader(arquivo_csv)
        for linha in leitor:
            nome_municipio = (linha.get("nome_municipio") or "").strip()
            if not nome_municipio:
                continue
            lote.append(
                {
                    "ano": _ler_int(linha.get("ano")),
                    "id_municipio": (linha.get("id_municipio") or "").strip() or None,
                    "nome_municipio": nome_municipio,
                    "quantidade_vinculos_ativos": _ler_int(linha.get("quantidade_vinculos_ativos")),
                    "quantidade_vinculos_clt": _ler_int(linha.get("quantidade_vinculos_clt")),
                    "quantidade_vinculos_estatutarios": _ler_int(linha.get("quantidade_vinculos_estatutarios")),
                    "natureza_juridica": (linha.get("natureza_juridica") or "").strip() or None,
                    "tamanho_estabelecimento": (linha.get("tamanho_estabelecimento") or "").strip() or None,
                    "tipo_estabelecimento": (linha.get("tipo_estabelecimento") or "").strip() or None,
                    "indicador_simples": (linha.get("indicador_simples") or "").strip() or None,
                    "cnae_2_subclasse": (linha.get("cnae_2_subclasse") or "").strip() or None,
                    "subsetor_ibge": (linha.get("subsetor_ibge") or "").strip() or None,
                    "cep": (linha.get("cep") or "").strip() or None,
                }
            )

            if len(lote) >= TAMANHO_LOTE:
                db.bulk_insert_mappings(Estabelecimento, lote)
                db.commit()
                total_importado += len(lote)
                lote = []

    if lote:
        db.bulk_insert_mappings(Estabelecimento, lote)
        db.commit()
        total_importado += len(lote)

    return total_importado


def seed_dados_mercado_trabalho():
    """
    Importa os CSVs externos de profissões e estabelecimentos (pasta
    `dados/`) na primeira inicialização, caso as tabelas correspondentes
    estejam vazias. Se os arquivos não existirem, não faz nada — a aplicação
    continua funcionando normalmente, apenas sem esses dados.
    """
    db = SessionLocal()
    try:
        if (db.query(func.count(Profissao.id)).scalar() or 0) == 0:
            _importar_profissoes_csv(
                db, DADOS_DIR / "profissoes_clt.csv", "total_trabalhadores_clt", "CLT"
            )
            _importar_profissoes_csv(
                db, DADOS_DIR / "profissoes_nao_clt.csv", "total_trabalhadores_nao_clt", "NAO_CLT"
            )
            _importar_profissoes_csv(
                db, DADOS_DIR / "profissoes_total.csv", "total_trabalhadores", "TOTAL"
            )

        if (db.query(func.count(Estabelecimento.id)).scalar() or 0) == 0:
            _importar_estabelecimentos_csv(db, DADOS_DIR / "estabelecimentos_rais.csv")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 5. APLICAÇÃO FASTAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RMS Data Hub",
    description="Extração, armazenamento e consulta de dados da Região Metropolitana de Sorocaba",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    seed_database()
    seed_dados_mercado_trabalho()


# ---------------------------------------------------------------------------
# 6. ENDPOINT: BUSCA (GET /api/busca)
# ---------------------------------------------------------------------------
@app.get("/api/busca", response_model=List[MunicipioSchema])
def buscar_municipios(
    q: str = Query(default="", description="Termo de busca pelo nome do município"),
    db: Session = Depends(get_db),
):
    """
    Busca municípios da RMS pelo nome, usando correspondência parcial
    (case-insensitive) via ILIKE/LIKE. Se `q` estiver vazio, retorna todos.
    """
    query = db.query(Municipio)

    if q:
        termo = f"%{q.strip()}%"
        # ilike funciona nativamente no PostgreSQL; SQLite trata LIKE como
        # case-insensitive por padrão para caracteres ASCII, então ilike()
        # do SQLAlchemy funciona corretamente em ambos os bancos.
        query = query.filter(Municipio.nome.ilike(termo))

    resultados = query.order_by(Municipio.nome.asc()).all()
    return resultados


# ---------------------------------------------------------------------------
# 7. ENDPOINT: SINCRONIZAÇÃO COM A API DO IBGE (POST /api/ibge/sincronizar)
# ---------------------------------------------------------------------------
IBGE_MUNICIPIOS_SP_URL = (
    "https://servicodados.ibge.gov.br/api/v1/localidades/estados/35/municipios"
)
# Tabela SIDRA 6579 = Estimativas de população; variável 9324 = população residente estimada
IBGE_POPULACAO_URL = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/-1/"
    "variaveis/9324?localidades=N6[{codigo}]"
)
# Tabela SIDRA 5938 = PIB dos municípios; variável 37 = PIB a preços correntes (mil reais)
IBGE_PIB_URL = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/-1/"
    "variaveis/37?localidades=N6[{codigo}]"
)


async def _buscar_codigo_ibge(client: httpx.AsyncClient, nome_municipio: str) -> Optional[str]:
    """Consulta a lista de municípios de SP no IBGE e retorna o código do
    município cujo nome bate (comparação normalizada, ignorando acentos e
    caixa) com `nome_municipio`."""
    resp = await client.get(IBGE_MUNICIPIOS_SP_URL, timeout=20.0)
    resp.raise_for_status()
    dados = resp.json()

    def normaliza(texto: str) -> str:
        import unicodedata

        texto = unicodedata.normalize("NFKD", texto)
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        return texto.strip().lower()

    alvo = normaliza(nome_municipio)
    for item in dados:
        if normaliza(item.get("nome", "")) == alvo:
            return str(item["id"])
    return None


async def _buscar_valor_sidra(client: httpx.AsyncClient, url_template: str, codigo: str) -> Optional[float]:
    """Faz a chamada à API de Agregados (SIDRA) do IBGE e extrai o valor
    numérico mais recente disponível para o município informado."""
    url = url_template.format(codigo=codigo)
    try:
        resp = await client.get(url, timeout=20.0)
        resp.raise_for_status()
        dados = resp.json()
        # Estrutura: [ { "id":..., "variavel":..., "resultados": [ { "series": [ { "serie": { "ano": valor } } ] } ] } ]
        serie = dados[0]["resultados"][0]["series"][0]["serie"]
        # Pega o valor do período mais recente disponível
        ultimo_ano = sorted(serie.keys())[-1]
        valor_bruto = serie[ultimo_ano]
        if valor_bruto in ("...", "-", None, ""):
            return None
        return float(valor_bruto)
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None


async def sincronizar_com_ibge(db: Session) -> dict:
    """
    Para cada município da RMS já cadastrado, busca o código IBGE
    correspondente e atualiza população estimada e PIB com os dados oficiais
    mais recentes disponíveis na base de Agregados (SIDRA) do IBGE.
    """
    municipios = db.query(Municipio).all()
    atualizados = []
    falhas = []

    async with httpx.AsyncClient() as client:
        for municipio in municipios:
            try:
                codigo = await _buscar_codigo_ibge(client, municipio.nome)
                if not codigo:
                    falhas.append(municipio.nome)
                    continue

                populacao, pib = await asyncio.gather(
                    _buscar_valor_sidra(client, IBGE_POPULACAO_URL, codigo),
                    _buscar_valor_sidra(client, IBGE_PIB_URL, codigo),
                )

                if populacao is not None:
                    municipio.populacao_estimada = int(populacao)
                if pib is not None:
                    municipio.pib = float(pib)

                municipio.data_atualizacao = datetime.now(timezone.utc)
                atualizados.append(municipio.nome)
            except Exception:
                falhas.append(municipio.nome)

    db.commit()
    return {"atualizados": atualizados, "falhas": falhas}


@app.post("/api/ibge/sincronizar")
async def sincronizar_ibge(db: Session = Depends(get_db)):
    """
    Consulta a API oficial do IBGE (localidades + agregados/SIDRA) e atualiza
    população estimada e PIB dos municípios da RMS já cadastrados no banco.
    """
    resultado = await sincronizar_com_ibge(db)
    return JSONResponse(
        content={
            "mensagem": "Sincronização com o IBGE concluída.",
            "total_atualizados": len(resultado["atualizados"]),
            "total_falhas": len(resultado["falhas"]),
            "municipios_atualizados": resultado["atualizados"],
            "municipios_com_falha": resultado["falhas"],
        }
    )


# ---------------------------------------------------------------------------
# 8. ENDPOINTS: MERCADO DE TRABALHO (Profissões e Estabelecimentos)
# ---------------------------------------------------------------------------
@app.get("/api/profissoes/municipios", response_model=List[str])
def listar_municipios_profissoes(db: Session = Depends(get_db)):
    """Lista (ordenada) os municípios presentes na base de profissões, para
    alimentar o filtro por município na interface."""
    linhas = (
        db.query(Profissao.nome_municipio)
        .distinct()
        .order_by(Profissao.nome_municipio.asc())
        .all()
    )
    return [linha[0] for linha in linhas]


@app.get("/api/profissoes", response_model=List[ProfissaoSchema])
def buscar_profissoes(
    q: str = Query(default="", description="Termo de busca pelo nome da profissão"),
    municipio: str = Query(default="", description="Filtra por nome exato do município"),
    tipo: str = Query(
        default="",
        description="Filtra por tipo de vínculo: CLT, NAO_CLT ou TOTAL",
    ),
    limite: int = Query(default=200, ge=1, le=1000, description="Máximo de resultados"),
    db: Session = Depends(get_db),
):
    """
    Busca profissões por nome (ILIKE), com filtros opcionais de município e
    tipo de vínculo, ordenadas pelo número de trabalhadores (maior primeiro).
    """
    query = db.query(Profissao)

    if q:
        query = query.filter(Profissao.nome_profissao.ilike(f"%{q.strip()}%"))
    if municipio:
        query = query.filter(Profissao.nome_municipio == municipio.strip())
    if tipo:
        query = query.filter(Profissao.tipo_vinculo == tipo.strip().upper())

    resultados = (
        query.order_by(Profissao.total_trabalhadores.desc().nullslast())
        .limit(limite)
        .all()
    )
    return resultados


@app.get("/api/estabelecimentos/resumo", response_model=List[EstabelecimentoResumoSchema])
def resumo_estabelecimentos(
    municipio: str = Query(default="", description="Filtra por nome exato do município"),
    db: Session = Depends(get_db),
):
    """
    Retorna, por município, o total de estabelecimentos e a soma dos
    vínculos empregatícios (ativos, CLT e estatutários) — agregado a partir
    dos microdados da RAIS 2021.
    """
    query = db.query(
        Estabelecimento.nome_municipio.label("nome_municipio"),
        func.count(Estabelecimento.id).label("total_estabelecimentos"),
        func.coalesce(func.sum(Estabelecimento.quantidade_vinculos_ativos), 0).label(
            "total_vinculos_ativos"
        ),
        func.coalesce(func.sum(Estabelecimento.quantidade_vinculos_clt), 0).label(
            "total_vinculos_clt"
        ),
        func.coalesce(func.sum(Estabelecimento.quantidade_vinculos_estatutarios), 0).label(
            "total_vinculos_estatutarios"
        ),
    ).group_by(Estabelecimento.nome_municipio)

    if municipio:
        query = query.filter(Estabelecimento.nome_municipio == municipio.strip())

    resultados = query.order_by(func.count(Estabelecimento.id).desc()).all()
    return [
        EstabelecimentoResumoSchema(
            nome_municipio=linha.nome_municipio,
            total_estabelecimentos=linha.total_estabelecimentos,
            total_vinculos_ativos=linha.total_vinculos_ativos,
            total_vinculos_clt=linha.total_vinculos_clt,
            total_vinculos_estatutarios=linha.total_vinculos_estatutarios,
        )
        for linha in resultados
    ]


# ---------------------------------------------------------------------------
# 9. FRONTEND EMBUTIDO (HTML + CSS + JS via HTMLResponse)
# ---------------------------------------------------------------------------
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>RMS Data Hub — Região Metropolitana de Sorocaba</title>
<style>
  :root {
    --cor-primaria: #1e5f8c;
    --cor-primaria-escura: #123f5e;
    --cor-fundo: #f2f5f8;
    --cor-cartao: #ffffff;
    --cor-texto: #1f2933;
    --cor-texto-suave: #62748a;
    --cor-badge: #e6f0fa;
    --sombra: 0 4px 14px rgba(18, 63, 94, 0.08);
    --sombra-hover: 0 8px 24px rgba(18, 63, 94, 0.16);
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--cor-fundo);
    color: var(--cor-texto);
  }

  header {
    background: linear-gradient(135deg, var(--cor-primaria), var(--cor-primaria-escura));
    color: #fff;
    padding: 36px 20px 60px;
    text-align: center;
  }

  header h1 {
    margin: 0 0 6px;
    font-size: 1.9rem;
  }

  header p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.95rem;
  }

  .container {
    max-width: 1100px;
    margin: -36px auto 40px;
    padding: 0 20px;
  }

  .busca-wrapper {
    background: var(--cor-cartao);
    border-radius: 14px;
    box-shadow: var(--sombra);
    padding: 18px 20px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }

  #campo-busca {
    flex: 1;
    min-width: 220px;
    padding: 14px 18px;
    font-size: 1rem;
    border: 1px solid #d7dee6;
    border-radius: 10px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  #campo-busca:focus {
    border-color: var(--cor-primaria);
    box-shadow: 0 0 0 3px rgba(30, 95, 140, 0.15);
  }

  #btn-sincronizar {
    padding: 14px 20px;
    font-size: 0.9rem;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    background: var(--cor-primaria);
    color: #fff;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.2s, transform 0.1s;
  }

  #btn-sincronizar:hover { background: var(--cor-primaria-escura); }
  #btn-sincronizar:active { transform: scale(0.98); }
  #btn-sincronizar:disabled { opacity: 0.6; cursor: not-allowed; }

  #status-msg {
    margin: 10px 4px 0;
    font-size: 0.85rem;
    color: var(--cor-texto-suave);
    min-height: 18px;
  }

  .contador {
    margin: 22px 4px 12px;
    font-size: 0.9rem;
    color: var(--cor-texto-suave);
  }

  .grid-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 18px;
  }

  .card {
    background: var(--cor-cartao);
    border-radius: 14px;
    padding: 20px;
    box-shadow: var(--sombra);
    transition: transform 0.15s, box-shadow 0.15s;
    border: 1px solid #eef1f5;
  }

  .card:hover {
    transform: translateY(-3px);
    box-shadow: var(--sombra-hover);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
  }

  .card-nome {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--cor-primaria-escura);
    margin: 0;
  }

  .badge-rms {
    background: var(--cor-badge);
    color: var(--cor-primaria);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 4px 9px;
    border-radius: 999px;
    white-space: nowrap;
  }

  .card-info {
    font-size: 0.87rem;
    color: var(--cor-texto-suave);
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .card-info strong {
    color: var(--cor-texto);
  }

  .vazio {
    text-align: center;
    padding: 60px 20px;
    color: var(--cor-texto-suave);
  }

  footer {
    text-align: center;
    padding: 24px;
    font-size: 0.8rem;
    color: var(--cor-texto-suave);
  }

  /* --- Abas --- */
  .abas {
    display: flex;
    gap: 8px;
    margin: 18px 4px 0;
    flex-wrap: wrap;
  }

  .aba-botao {
    padding: 10px 18px;
    border: none;
    border-radius: 10px 10px 0 0;
    background: #e3e9ef;
    color: var(--cor-texto-suave);
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
  }

  .aba-botao.ativa {
    background: var(--cor-cartao);
    color: var(--cor-primaria-escura);
    box-shadow: var(--sombra);
  }

  .painel-aba { display: none; }
  .painel-aba.ativo { display: block; }

  /* --- Filtros (mercado de trabalho) --- */
  .filtros-wrapper {
    background: var(--cor-cartao);
    border-radius: 0 14px 14px 14px;
    box-shadow: var(--sombra);
    padding: 18px 20px;
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }

  .filtros-wrapper select {
    padding: 12px 14px;
    font-size: 0.9rem;
    border: 1px solid #d7dee6;
    border-radius: 10px;
    background: #fff;
    color: var(--cor-texto);
    min-width: 170px;
  }

  .subtabs {
    display: flex;
    gap: 6px;
    margin: 18px 4px 10px;
  }

  .subtab-botao {
    padding: 7px 14px;
    border: 1px solid #d7dee6;
    border-radius: 999px;
    background: #fff;
    color: var(--cor-texto-suave);
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
  }

  .subtab-botao.ativa {
    background: var(--cor-primaria);
    border-color: var(--cor-primaria);
    color: #fff;
  }

  /* --- Tabelas --- */
  .tabela-wrapper {
    background: var(--cor-cartao);
    border-radius: 14px;
    box-shadow: var(--sombra);
    overflow-x: auto;
  }

  table.tabela-dados {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.87rem;
  }

  table.tabela-dados th {
    text-align: left;
    padding: 12px 16px;
    background: #f7f9fb;
    color: var(--cor-texto-suave);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid #eef1f5;
    white-space: nowrap;
  }

  table.tabela-dados td {
    padding: 11px 16px;
    border-bottom: 1px solid #f2f5f8;
    white-space: nowrap;
  }

  table.tabela-dados tr:last-child td { border-bottom: none; }
  table.tabela-dados tr:hover td { background: #f9fbfd; }

  .badge-vinculo {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    white-space: nowrap;
  }

  .badge-vinculo.CLT { background: #e6f0fa; color: #1e5f8c; }
  .badge-vinculo.NAO_CLT { background: #fdeee0; color: #b1560f; }
  .badge-vinculo.TOTAL { background: #e9f7ec; color: #1f7a3d; }

  @media (max-width: 480px) {
    header h1 { font-size: 1.5rem; }
    .busca-wrapper { flex-direction: column; align-items: stretch; }
    #btn-sincronizar { width: 100%; }
    .filtros-wrapper { flex-direction: column; align-items: stretch; }
  }
</style>
</head>
<body>

<header>
  <h1>🗺️ RMS Data Hub</h1>
  <p>Consulta de dados dos municípios e do mercado de trabalho da Região Metropolitana de Sorocaba</p>
</header>

<div class="container">

  <div class="abas">
    <button class="aba-botao ativa" data-aba="municipios">🏙️ Municípios</button>
    <button class="aba-botao" data-aba="mercado">💼 Mercado de Trabalho</button>
  </div>

  <!-- ABA: MUNICÍPIOS -->
  <div class="painel-aba ativo" id="painel-municipios">
    <div class="busca-wrapper" style="border-radius: 0 14px 14px 14px;">
      <input
        type="text"
        id="campo-busca"
        placeholder="Digite o nome de um município (ex.: Sorocaba, Itu, Votorantim...)"
        autocomplete="off"
      />
      <button id="btn-sincronizar">🔄 Sincronizar com IBGE</button>
    </div>
    <div id="status-msg"></div>

    <div class="contador" id="contador">Carregando municípios...</div>
    <div class="grid-cards" id="grid-resultados"></div>
  </div>

  <!-- ABA: MERCADO DE TRABALHO -->
  <div class="painel-aba" id="painel-mercado">

    <div class="subtabs">
      <button class="subtab-botao ativa" data-subtab="profissoes">Profissões e Salários</button>
      <button class="subtab-botao" data-subtab="estabelecimentos">Estabelecimentos (RAIS)</button>
    </div>

    <!-- SUBABA: PROFISSÕES -->
    <div class="painel-aba ativo" id="painel-profissoes">
      <div class="filtros-wrapper">
        <input
          type="text"
          id="campo-busca-profissao"
          placeholder="Buscar profissão (ex.: motorista, vendedor, professor...)"
          autocomplete="off"
          style="flex: 1; min-width: 220px; padding: 12px 14px; font-size: 0.9rem; border: 1px solid #d7dee6; border-radius: 10px; outline: none;"
        />
        <select id="filtro-municipio-profissao">
          <option value="">Todos os municípios</option>
        </select>
        <select id="filtro-tipo-vinculo">
          <option value="">Todos os vínculos</option>
          <option value="CLT">CLT</option>
          <option value="NAO_CLT">Não-CLT</option>
          <option value="TOTAL">Total</option>
        </select>
      </div>

      <div class="contador" id="contador-profissoes">Carregando dados de profissões...</div>
      <div class="tabela-wrapper">
        <table class="tabela-dados">
          <thead>
            <tr>
              <th>Profissão</th>
              <th>Município</th>
              <th>Vínculo</th>
              <th>Trabalhadores</th>
              <th>Salário médio</th>
            </tr>
          </thead>
          <tbody id="tbody-profissoes"></tbody>
        </table>
      </div>
    </div>

    <!-- SUBABA: ESTABELECIMENTOS -->
    <div class="painel-aba" id="painel-estabelecimentos">
      <div class="contador" id="contador-estabelecimentos">Carregando resumo de estabelecimentos...</div>
      <div class="tabela-wrapper">
        <table class="tabela-dados">
          <thead>
            <tr>
              <th>Município</th>
              <th>Estabelecimentos</th>
              <th>Vínculos ativos</th>
              <th>Vínculos CLT</th>
              <th>Vínculos estatutários</th>
            </tr>
          </thead>
          <tbody id="tbody-estabelecimentos"></tbody>
        </table>
      </div>
    </div>

  </div>

</div>

<footer>
  RMS Data Hub &middot; Municípios oficiais da RMS (IBGE) + dados de mercado de trabalho (RAIS / basedosdados.org)
</footer>

<script>
  const campoBusca = document.getElementById("campo-busca");
  const gridResultados = document.getElementById("grid-resultados");
  const contador = document.getElementById("contador");
  const btnSincronizar = document.getElementById("btn-sincronizar");
  const statusMsg = document.getElementById("status-msg");

  function formatarNumero(valor) {
    if (valor === null || valor === undefined) return "—";
    return Number(valor).toLocaleString("pt-BR");
  }

  function formatarPib(valor) {
    if (valor === null || valor === undefined) return "—";
    // valor armazenado em mil reais -> exibe em milhões de reais
    const emMilhoes = Number(valor) / 1000;
    return "R$ " + emMilhoes.toLocaleString("pt-BR", { maximumFractionDigits: 1 }) + " mi";
  }

  function renderizarCards(municipios) {
    gridResultados.innerHTML = "";

    if (!municipios || municipios.length === 0) {
      gridResultados.innerHTML = '<div class="vazio">Nenhum município encontrado para essa busca.</div>';
      contador.textContent = "0 municípios encontrados";
      return;
    }

    contador.textContent = municipios.length + " município(s) encontrado(s)";

    municipios.forEach((m) => {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `
        <div class="card-header">
          <h3 class="card-nome">${m.nome}</h3>
          <span class="badge-rms">RMS</span>
        </div>
        <div class="card-info">
          <span><strong>Estado:</strong> ${m.estado}</span>
          <span><strong>População:</strong> ${formatarNumero(m.populacao_estimada)}</span>
          <span><strong>PIB:</strong> ${formatarPib(m.pib)}</span>
        </div>
      `;
      gridResultados.appendChild(card);
    });
  }

  async function buscarMunicipios(termo) {
    try {
      const resposta = await fetch("/api/busca?q=" + encodeURIComponent(termo));
      const dados = await resposta.json();
      renderizarCards(dados);
    } catch (erro) {
      gridResultados.innerHTML = '<div class="vazio">Erro ao consultar os dados. Tente novamente.</div>';
      console.error(erro);
    }
  }

  // Busca instantânea conforme o usuário digita
  campoBusca.addEventListener("input", (evento) => {
    buscarMunicipios(evento.target.value);
  });

  btnSincronizar.addEventListener("click", async () => {
    btnSincronizar.disabled = true;
    btnSincronizar.textContent = "Sincronizando...";
    statusMsg.textContent = "Consultando a API do IBGE, isso pode levar alguns segundos...";

    try {
      const resposta = await fetch("/api/ibge/sincronizar", { method: "POST" });
      const dados = await resposta.json();
      statusMsg.textContent =
        `✅ ${dados.total_atualizados} município(s) atualizado(s)` +
        (dados.total_falhas > 0 ? `, ${dados.total_falhas} com falha.` : ".");
      buscarMunicipios(campoBusca.value);
    } catch (erro) {
      statusMsg.textContent = "❌ Erro ao sincronizar com o IBGE.";
      console.error(erro);
    } finally {
      btnSincronizar.disabled = false;
      btnSincronizar.textContent = "🔄 Sincronizar com IBGE";
    }
  });

  // Carrega a lista completa ao abrir a página
  buscarMunicipios("");

  // -------------------------------------------------------------------
  // Navegação entre abas / subabas
  // -------------------------------------------------------------------
  document.querySelectorAll(".aba-botao").forEach((botao) => {
    botao.addEventListener("click", () => {
      document.querySelectorAll(".aba-botao").forEach((b) => b.classList.remove("ativa"));
      botao.classList.add("ativa");

      const alvo = botao.dataset.aba;
      document.getElementById("painel-municipios").classList.toggle("ativo", alvo === "municipios");
      document.getElementById("painel-mercado").classList.toggle("ativo", alvo === "mercado");

      if (alvo === "mercado" && !mercadoCarregado) {
        inicializarMercadoTrabalho();
      }
    });
  });

  document.querySelectorAll(".subtab-botao").forEach((botao) => {
    botao.addEventListener("click", () => {
      document.querySelectorAll(".subtab-botao").forEach((b) => b.classList.remove("ativa"));
      botao.classList.add("ativa");

      const alvo = botao.dataset.subtab;
      document.getElementById("painel-profissoes").classList.toggle("ativo", alvo === "profissoes");
      document.getElementById("painel-estabelecimentos").classList.toggle("ativo", alvo === "estabelecimentos");
    });
  });

  // -------------------------------------------------------------------
  // Mercado de Trabalho: Profissões e Salários
  // -------------------------------------------------------------------
  let mercadoCarregado = false;

  const campoBuscaProfissao = document.getElementById("campo-busca-profissao");
  const filtroMunicipioProfissao = document.getElementById("filtro-municipio-profissao");
  const filtroTipoVinculo = document.getElementById("filtro-tipo-vinculo");
  const tbodyProfissoes = document.getElementById("tbody-profissoes");
  const contadorProfissoes = document.getElementById("contador-profissoes");
  const tbodyEstabelecimentos = document.getElementById("tbody-estabelecimentos");
  const contadorEstabelecimentos = document.getElementById("contador-estabelecimentos");

  const RÓTULO_VINCULO = { CLT: "CLT", NAO_CLT: "Não-CLT", TOTAL: "Total" };

  function formatarSalario(valor) {
    if (valor === null || valor === undefined) return "—";
    return "R$ " + Number(valor).toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  async function inicializarMercadoTrabalho() {
    mercadoCarregado = true;
    await carregarMunicipiosProfissoes();
    await buscarProfissoes();
    await carregarResumoEstabelecimentos();
  }

  async function carregarMunicipiosProfissoes() {
    try {
      const resposta = await fetch("/api/profissoes/municipios");
      const municipios = await resposta.json();
      municipios.forEach((nome) => {
        const opcao = document.createElement("option");
        opcao.value = nome;
        opcao.textContent = nome;
        filtroMunicipioProfissao.appendChild(opcao);
      });
    } catch (erro) {
      console.error(erro);
    }
  }

  async function buscarProfissoes() {
    const params = new URLSearchParams({
      q: campoBuscaProfissao.value.trim(),
      municipio: filtroMunicipioProfissao.value,
      tipo: filtroTipoVinculo.value,
    });

    contadorProfissoes.textContent = "Buscando...";
    try {
      const resposta = await fetch("/api/profissoes?" + params.toString());
      const dados = await resposta.json();
      renderizarProfissoes(dados);
    } catch (erro) {
      tbodyProfissoes.innerHTML = "";
      contadorProfissoes.textContent = "Erro ao consultar os dados.";
      console.error(erro);
    }
  }

  function renderizarProfissoes(linhas) {
    tbodyProfissoes.innerHTML = "";

    if (!linhas || linhas.length === 0) {
      contadorProfissoes.textContent = "Nenhum resultado encontrado.";
      return;
    }

    contadorProfissoes.textContent = linhas.length + " resultado(s) (máx. 200 exibidos, ordenados por nº de trabalhadores)";

    linhas.forEach((linha) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${linha.nome_profissao}</td>
        <td>${linha.nome_municipio}</td>
        <td><span class="badge-vinculo ${linha.tipo_vinculo}">${RÓTULO_VINCULO[linha.tipo_vinculo] || linha.tipo_vinculo}</span></td>
        <td>${formatarNumero(linha.total_trabalhadores)}</td>
        <td>${formatarSalario(linha.salario_medio)}</td>
      `;
      tbodyProfissoes.appendChild(tr);
    });
  }

  campoBuscaProfissao.addEventListener("input", buscarProfissoes);
  filtroMunicipioProfissao.addEventListener("change", buscarProfissoes);
  filtroTipoVinculo.addEventListener("change", buscarProfissoes);

  // -------------------------------------------------------------------
  // Mercado de Trabalho: Estabelecimentos (resumo por município)
  // -------------------------------------------------------------------
  async function carregarResumoEstabelecimentos() {
    contadorEstabelecimentos.textContent = "Carregando...";
    try {
      const resposta = await fetch("/api/estabelecimentos/resumo");
      const linhas = await resposta.json();
      renderizarEstabelecimentos(linhas);
    } catch (erro) {
      tbodyEstabelecimentos.innerHTML = "";
      contadorEstabelecimentos.textContent = "Erro ao consultar os dados.";
      console.error(erro);
    }
  }

  function renderizarEstabelecimentos(linhas) {
    tbodyEstabelecimentos.innerHTML = "";

    if (!linhas || linhas.length === 0) {
      contadorEstabelecimentos.textContent =
        "Nenhum dado de estabelecimentos importado (arquivo dados/estabelecimentos_rais.csv não encontrado).";
      return;
    }

    contadorEstabelecimentos.textContent = linhas.length + " município(s) com dados de estabelecimentos (RAIS 2021)";

    linhas.forEach((linha) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${linha.nome_municipio}</td>
        <td>${formatarNumero(linha.total_estabelecimentos)}</td>
        <td>${formatarNumero(linha.total_vinculos_ativos)}</td>
        <td>${formatarNumero(linha.total_vinculos_clt)}</td>
        <td>${formatarNumero(linha.total_vinculos_estatutarios)}</td>
      `;
      tbodyEstabelecimentos.appendChild(tr);
    });
  }
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def pagina_inicial():
    """Serve a página HTML da interface de busca."""
    return HTMLResponse(content=FRONTEND_HTML)


# ---------------------------------------------------------------------------
# 10. EXECUÇÃO DIRETA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # PORT é definida automaticamente por serviços de hospedagem como o
    # Render; localmente, sem essa variável, cai no padrão 8000.
    porta = int(os.environ.get("PORT", 8000))
    # reload=True é ótimo em desenvolvimento local, mas deve ficar desligado
    # em produção — ativa automaticamente só quando não há PORT definida
    # (ou seja, quando você roda "python app.py" na sua máquina).
    modo_local = "PORT" not in os.environ
    uvicorn.run("app:app", host="0.0.0.0", port=porta, reload=modo_local)
