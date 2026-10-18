from reportlab.pdfgen import canvas

textos = [
    "Identificação do medicamento: {med}",
    "Indicação: Tratamento de dor e febre.",
    "Como este medicamento funciona: Atua como analgésico.",
    "Contraindicação: Hipersensibilidade ao componente.",
    "Posologia: Tomar 500mg a cada 6 horas."
]

for med in ["dipirona", "paracetamol"]:
    c = canvas.Canvas(f"{med}.pdf")
    y = 800
    for linha in textos:
        c.drawString(50, y, linha.format(med=med))
        y -= 20
    c.save()
