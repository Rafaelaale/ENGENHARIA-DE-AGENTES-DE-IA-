from reportlab.pdfgen import canvas

textos = {
    "politica_ferias.pdf": ["Política de Férias", "O funcionário tem direito a 30 dias de férias por ano.", "É necessário aviso prévio de 30 dias para solicitação."],
    "politica_home_office.pdf": ["Política de Home Office", "O trabalho remoto é permitido até 3 vezes na semana.", "A empresa fornece notebook e ajuda de custo para internet."],
    "codigo_conduta.pdf": ["Código de Conduta", "Comportamentos inadequados incluem: assédio, discriminação e desrespeito.", "A ética e o respeito mútuo são fundamentais no ambiente de trabalho."]
}

for arquivo, linhas in textos.items():
    c = canvas.Canvas(arquivo)
    y = 800
    for linha in linhas:
        c.drawString(50, y, linha)
        y -= 20
    c.save()
