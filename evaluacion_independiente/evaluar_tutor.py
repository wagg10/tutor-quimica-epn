#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluar_tutor.py
================
Evaluación por lotes del tutor académico de Química (Fundamentos de Química, EPN).

Qué hace
--------
1. Lee un conjunto de prueba en formato JSON Lines (un ejercicio por línea).
2. Envía cada ejercicio al modelo, junto con la skill pedagógica como
   instrucción de sistema.
3. Extrae el resultado final del apartado "Resultado obtenido" de la respuesta.
4. Califica automáticamente:
      - Resultado numérico: compara contra el valor esperado con tolerancia.
      - Estructura: verifica los ocho apartados (ejercicios) o los cuatro
        apartados conceptuales (definición, explicación, ejemplo, importancia).
      - Contenido no solicitado: detecta código, gráficos o etiquetas internas.
5. Agrega el acierto por eje temático del sílabo y en total, y exporta las
   tablas listas para el capítulo de resultados.

Alcance de la calificación automática (importante para la defensa)
------------------------------------------------------------------
- Los ejercicios de resultado NUMÉRICO se califican de forma automática y
  objetiva (resultado correcto sí/no, dentro de la tolerancia declarada).
- Los ejercicios SIMBÓLICOS (configuración electrónica, nomenclatura, orden
  de propiedades) y las preguntas CONCEPTUALES se marcan como
  "revision_manual": el script guarda la respuesta completa y verifica solo la
  estructura, porque su corrección admite varios formatos válidos y no puede
  automatizarse sin riesgo de error. Esos se califican con la rúbrica a mano.
- La distinción fina entre error aritmético y error por arrastre de datos no
  se detecta de forma fiable en lote. El script marca el ejercicio como
  "resultado_incorrecto" y lo deja para clasificación manual.

Rúbrica sugerida (documéntala como tabla en la tesis)
-----------------------------------------------------
Por cada ejercicio se registran cuatro criterios:
  C1. Resultado final correcto      -> dentro de la tolerancia declarada.
  C2. Estructura completa           -> los 8 apartados (o 4 conceptuales).
  C3. Procedimiento sin errores     -> revisión manual del desarrollo.
  C4. Sin datos inventados          -> no introduce valores ajenos al enunciado.
Métrica principal reportada: % de C1 (resultado correcto) por eje y total.
Métricas secundarias: % de C2 (estructura) y tipología de errores.

Backends
--------
--backend api  (por defecto): consume un servidor local compatible con OpenAI,
               como llama-server sirviendo el modelo GGUF de Phi-4.
--backend peft            : carga un modelo base de Hugging Face y le acopla el
               adaptador QLoRA con PeftModel.from_pretrained. Sirve para evaluar
               Qwen2.5-14B-Instruct y DeepSeek-R1-Distill-Qwen-14B sin
               reentrenar. Requiere el entorno de entrenamiento (torch,
               transformers, peft, bitsandbytes) y una GPU.

Ejemplos de uso
---------------
# Phi-4 (GGUF vía llama-server), con la skill:
python evaluar_tutor.py \
    --conjunto conjunto_prueba_quimica.jsonl \
    --skill tutor-quimica-epn-runtime.md \
    --backend api \
    --base-url http://127.0.0.1:8081/v1 \
    --modelo phi4_quimica_v2_BF16.gguf \
    --salida resultados_phi4

# El mismo Phi-4 sin skill (para reproducir la comparación con/sin skill):
python evaluar_tutor.py --conjunto conjunto_prueba_quimica.jsonl \
    --sin-skill --backend api --salida resultados_phi4_sin_skill

# Qwen2.5 con su adaptador (comparación, sin reentrenar):
python evaluar_tutor.py \
    --conjunto conjunto_prueba_quimica.jsonl \
    --skill tutor-quimica-epn-runtime.md \
    --backend peft \
    --modelo-base Qwen/Qwen2.5-14B-Instruct \
    --adaptador ./modelo_tutor_quimica_qwen25_14b_v2 \
    --carga 8bit --salida resultados_qwen
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path


# --------------------------------------------------------------------------- #
# Utilidades de texto y extracción de números
# --------------------------------------------------------------------------- #

def sin_acentos(texto: str) -> str:
    """Devuelve el texto sin tildes, para búsquedas robustas."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def limpiar_latex(texto: str) -> str:
    """Elimina el ruido de LaTeX para poder leer los números del resultado."""
    t = texto
    # Quitar unidades escritas como \text{...} o \mathrm{...}
    t = re.sub(r"\\text\{[^{}]*\}", " ", t)
    t = re.sub(r"\\mathrm\{[^{}]*\}", " ", t)
    # Separador decimal en LaTeX ({,} o {.}) antes de eliminar las llaves
    t = t.replace("{,}", ",").replace("{.}", ".")
    # Separadores finos de LaTeX
    for s in [r"\,", r"\;", r"\:", r"\!", r"\ "]:
        t = t.replace(s, " ")
    # Símbolo de grado y llaves
    t = re.sub(r"\^\{?\\circ\}?", " ", t)
    t = t.replace("\\circ", " ")
    # Cualquier otro comando \algo
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    for s in ["$", "\\(", "\\)", "\\[", "\\]", "{", "}"]:
        t = t.replace(s, " ")
    # Coma decimal -> punto, solo cuando está entre dígitos
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    return t


PATRON_NUMERO = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def quitar_encabezados(texto: str) -> str:
    """Elimina encabezados (### 8. ..., **8. ...**, '8. ' inicial) para que el
    número del título del siguiente apartado no se confunda con el resultado."""
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s+.*$", " ", texto)   # ### Encabezado
    t = re.sub(r"\*\*\s*\d+\s*\..*?\*\*", " ", t)          # **8. ...**
    t = re.sub(r"(?m)^\s*\d+\s*\.\s", " ", t)              # "8. " al inicio de línea
    return t


def extraer_numeros(texto: str):
    """Devuelve todos los números presentes en un texto ya limpio de LaTeX."""
    return [float(x) for x in PATRON_NUMERO.findall(limpiar_latex(quitar_encabezados(texto)))]


def indice_primero(texto_min: str, anclas) -> int:
    """Primer índice donde aparece alguna de las anclas (texto en minúsculas)."""
    for a in anclas:
        i = texto_min.find(a)
        if i != -1:
            return i
    return -1


# Anclas de los ocho apartados del ejercicio (minúsculas, sin tildes).
ANCLAS_EJERCICIO = [
    ["fundamento qu"],                 # 1. Fundamento químico del problema
    ["informacion proporcionada", "informacion"],  # 2. Información proporcionada
    ["lo que se debe determinar", "lo que se debe"],  # 3.
    ["relaciones qu"],                 # 4. Relaciones químicas y expresiones
    ["aplicacion de los datos", "aplicacion"],  # 5.
    ["resolucion paso a paso", "resolucion"],   # 6.
    ["resultado obtenido", "resultado final"],  # 7.
    ["conclusion"],                    # 8. Conclusión
]

# Anclas de la respuesta conceptual (cuatro apartados).
ANCLAS_CONCEPTUAL = [
    ["definicion"],
    ["explicacion"],
    ["ejemplo"],
    ["importancia"],
]

# Señales de contenido que la skill prohíbe (código, gráficos, razonamiento interno).
SENALES_NO_SOLICITADAS = [
    "```", "<think>", "</think>", "<|", "assistantfinal",
    "matplotlib", "import numpy", "plt.",
]


def apartados_presentes(respuesta: str, anclas_grupo):
    """Cuenta cuántos apartados de la estructura aparecen en la respuesta."""
    texto_min = sin_acentos(respuesta.lower())
    presentes = 0
    faltantes = []
    for i, anclas in enumerate(anclas_grupo, start=1):
        if indice_primero(texto_min, anclas) != -1:
            presentes += 1
        else:
            faltantes.append(i)
    return presentes, faltantes


def seccion_resultado(respuesta: str) -> str:
    """Extrae el texto del apartado 'Resultado obtenido' hasta 'Conclusión'."""
    texto_min = sin_acentos(respuesta.lower())
    ini = indice_primero(texto_min, ["resultado obtenido", "resultado final"])
    if ini == -1:
        return ""  # no se encontró el apartado
    fin = texto_min.find("conclusi", ini + 1)
    if fin == -1:
        fin = len(respuesta)
    return respuesta[ini:fin]


def tiene_contenido_no_solicitado(respuesta: str) -> bool:
    r = respuesta.lower()
    return any(s in r for s in SENALES_NO_SOLICITADAS)


def dentro_tolerancia(valor, esperado, tol_abs, tol_rel) -> bool:
    """Verifica si un valor cae dentro de la tolerancia absoluta o relativa."""
    if valor is None:
        return False
    if tol_abs is not None and abs(valor - esperado) <= tol_abs:
        return True
    if tol_rel is not None and esperado != 0 and abs(valor - esperado) <= abs(esperado) * tol_rel:
        return True
    if esperado == 0 and tol_abs is not None:
        return abs(valor) <= tol_abs
    return False


# --------------------------------------------------------------------------- #
# Backends de inferencia
# --------------------------------------------------------------------------- #

def responder_api(enunciado, sistema, base_url, modelo, temperatura, max_tokens):
    """Consulta un servidor local compatible con OpenAI (llama-server / LM Studio)."""
    import requests  # import local para no exigirlo en modo peft

    mensajes = []
    if sistema:
        mensajes.append({"role": "system", "content": sistema})
    mensajes.append({"role": "user", "content": enunciado})

    url = base_url.rstrip("/") + "/chat/completions"
    cuerpo = {
        "model": modelo,
        "messages": mensajes,
        "temperature": temperatura,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(url, json=cuerpo, timeout=600)
    r.raise_for_status()
    datos = r.json()
    return datos["choices"][0]["message"]["content"]


def cargar_peft(modelo_base, adaptador, carga):
    """Carga el modelo base con su adaptador QLoRA (sin reentrenar)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(modelo_base)

    kwargs = {"device_map": "auto"}
    if carga == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif carga == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:  # bf16
        kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(modelo_base, **kwargs)
    modelo = PeftModel.from_pretrained(base, adaptador)
    modelo.eval()
    return modelo, tok


def responder_peft(enunciado, sistema, modelo, tok, max_tokens):
    import torch

    mensajes = []
    if sistema:
        mensajes.append({"role": "system", "content": sistema})
    mensajes.append({"role": "user", "content": enunciado})

    entrada = tok.apply_chat_template(
        mensajes,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    entrada = {k: v.to(modelo.device) for k, v in entrada.items()}

    with torch.no_grad():
        salida = modelo.generate(
            **entrada,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    prompt_len = entrada["input_ids"].shape[1]
    generado = salida[0][prompt_len:]
    return tok.decode(generado, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #

EJES_SILABO = [
    "estructura_atomica", "configuracion_electronica", "propiedades_periodicas",
    "estados_oxidacion", "reacciones_quimicas", "estequiometria",
]


def cargar_conjunto(ruta):
    items = []
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea or linea.startswith("//"):
                continue
            try:
                items.append(json.loads(linea))
            except json.JSONDecodeError as e:
                print(f"[aviso] Línea {n} ignorada por error JSON: {e}")
    return items


def clasificar(item, respuesta):
    """Aplica la calificación automática a una respuesta y devuelve un dict."""
    tipo = item.get("tipo", "ejercicio_numerico")
    esperado = item.get("resultado_esperado")
    unidad = item.get("unidad", "")
    tol_abs = item.get("tolerancia_abs")
    tol_rel = item.get("tolerancia_rel", 0.005)

    anclas = ANCLAS_CONCEPTUAL if tipo == "conceptual" else ANCLAS_EJERCICIO
    n_apartados, faltantes = apartados_presentes(respuesta, anclas)
    total_apartados = len(anclas)
    estructura_completa = (n_apartados == total_apartados)
    contenido_extra = tiene_contenido_no_solicitado(respuesta)

    seccion = seccion_resultado(respuesta)
    numeros = extraer_numeros(seccion) if seccion else []
    num_detectado = numeros[-1] if numeros else None

    if tipo == "ejercicio_numerico":
        if num_detectado is None:
            resultado_correcto = False
            clasificacion = "sin_resultado_detectado"
        else:
            resultado_correcto = dentro_tolerancia(num_detectado, esperado, tol_abs, tol_rel)
            if resultado_correcto and estructura_completa and not contenido_extra:
                clasificacion = "correcto"
            elif resultado_correcto:
                clasificacion = "correcto_con_formato"
            else:
                clasificacion = "resultado_incorrecto"
    else:
        resultado_correcto = None  # requiere rúbrica manual
        clasificacion = "revision_manual"

    return {
        "id": item.get("id", ""),
        "eje": item.get("eje", ""),
        "tipo": tipo,
        "esperado": esperado,
        "unidad": unidad,
        "num_detectado": num_detectado,
        "todos_los_numeros": ";".join(str(x) for x in numeros),
        "resultado_correcto": resultado_correcto,
        "apartados_detectados": f"{n_apartados}/{total_apartados}",
        "apartados_faltantes": ";".join(str(x) for x in faltantes),
        "estructura_completa": estructura_completa,
        "contenido_no_solicitado": contenido_extra,
        "clasificacion": clasificacion,
    }


def escribir_csv(ruta, filas, columnas):
    import csv
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        for fila in filas:
            w.writerow({c: fila.get(c, "") for c in columnas})


def agregar_por_eje(resultados):
    """Calcula acierto y estructura por eje temático y en total."""
    resumen = {}
    for eje in EJES_SILABO + ["(otros)"]:
        resumen[eje] = {"n": 0, "n_num": 0, "n_correcto": 0, "n_estructura": 0}

    for r in resultados:
        eje = r["eje"] if r["eje"] in resumen else "(otros)"
        d = resumen[eje]
        d["n"] += 1
        if r["estructura_completa"]:
            d["n_estructura"] += 1
        if r["tipo"] == "ejercicio_numerico":
            d["n_num"] += 1
            if r["clasificacion"] in ("correcto", "correcto_con_formato"):
                d["n_correcto"] += 1
    return resumen


def pct(a, b):
    return f"{(100.0 * a / b):.1f} %" if b else "-"


def escribir_resumen_md(ruta, resultados, resumen, meta):
    lineas = []
    lineas.append("# Resumen de la evaluación del tutor de Química\n")
    lineas.append(f"- Modelo evaluado: {meta['modelo']}")
    lineas.append(f"- Skill aplicada: {meta['skill']}")
    lineas.append(f"- Ejercicios evaluados: {len(resultados)}")
    lineas.append(f"- Fecha: {time.strftime('%Y-%m-%d %H:%M')}\n")

    lineas.append("## Acierto por eje temático\n")
    lineas.append("| Eje del sílabo | Ejercicios | Numéricos | Resultado correcto | % acierto | % estructura completa |")
    lineas.append("|---|---|---|---|---|---|")
    tot = {"n": 0, "n_num": 0, "n_correcto": 0, "n_estructura": 0}
    for eje in EJES_SILABO + ["(otros)"]:
        d = resumen[eje]
        if d["n"] == 0:
            continue
        for k in tot:
            tot[k] += d[k]
        lineas.append(
            f"| {eje} | {d['n']} | {d['n_num']} | {d['n_correcto']} | "
            f"{pct(d['n_correcto'], d['n_num'])} | {pct(d['n_estructura'], d['n'])} |"
        )
    lineas.append(
        f"| **Total** | **{tot['n']}** | **{tot['n_num']}** | **{tot['n_correcto']}** | "
        f"**{pct(tot['n_correcto'], tot['n_num'])}** | **{pct(tot['n_estructura'], tot['n'])}** |\n"
    )

    lineas.append("## Tipología de resultados (calificación automática)\n")
    conteo = {}
    for r in resultados:
        conteo[r["clasificacion"]] = conteo.get(r["clasificacion"], 0) + 1
    lineas.append("| Clasificación | Ejercicios |")
    lineas.append("|---|---|")
    etiquetas = {
        "correcto": "Correcto (resultado y estructura)",
        "correcto_con_formato": "Resultado correcto, estructura o formato incompleto",
        "resultado_incorrecto": "Resultado incorrecto (revisar: aritmético o arrastre)",
        "sin_resultado_detectado": "Sin resultado detectado (revisar extracción)",
        "revision_manual": "Simbólico o conceptual (rúbrica manual)",
    }
    for clave, texto in etiquetas.items():
        if clave in conteo:
            lineas.append(f"| {texto} | {conteo[clave]} |")
    lineas.append("")
    lineas.append("> Nota: el acierto automático solo aplica a ejercicios de resultado numérico. "
                  "Los simbólicos y conceptuales, y todo caso marcado como 'resultado_incorrecto', "
                  "se revisan con la rúbrica antes de reportarse.")

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))


def main():
    ap = argparse.ArgumentParser(description="Evaluación por lotes del tutor de Química.")
    ap.add_argument("--conjunto", required=True, help="Conjunto de prueba en JSONL.")
    ap.add_argument("--skill", help="Archivo .md de la skill pedagógica (instrucción de sistema).")
    ap.add_argument("--sin-skill", action="store_true", help="Ejecuta sin skill.")
    ap.add_argument("--salida", default="resultados_evaluacion", help="Carpeta de salida.")
    ap.add_argument("--limite", type=int, default=0, help="Evalúa solo los primeros N (0 = todos).")

    ap.add_argument("--backend", choices=["api", "peft"], default="api")
    # Backend API
    ap.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--modelo", default="phi4_quimica_v2_BF16.gguf")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1200)
    # Backend PEFT
    ap.add_argument("--modelo-base", help="Modelo base de Hugging Face.")
    ap.add_argument("--adaptador", help="Carpeta del adaptador QLoRA.")
    ap.add_argument("--carga", choices=["4bit", "8bit", "bf16"], default="8bit")

    args = ap.parse_args()

    # Skill
    sistema = ""
    if not args.sin_skill:
        if not args.skill:
            print("[aviso] No se indicó --skill. Se ejecuta sin instrucción de sistema. "
                  "Usa --sin-skill para hacerlo explícito o pasa --skill <archivo>.")
        else:
            sistema = Path(args.skill).read_text(encoding="utf-8")

    items = cargar_conjunto(args.conjunto)
    if args.limite:
        items = items[:args.limite]
    if not items:
        print("No hay ejercicios que evaluar. Revisa el conjunto de prueba.")
        sys.exit(1)

    salida = Path(args.salida)
    (salida / "respuestas").mkdir(parents=True, exist_ok=True)

    # Preparar backend
    modelo_peft = tok_peft = None
    if args.backend == "peft":
        if not args.modelo_base or not args.adaptador:
            print("En modo peft debes pasar --modelo-base y --adaptador.")
            sys.exit(1)
        print(f"Cargando {args.modelo_base} + adaptador ({args.carga})...")
        modelo_peft, tok_peft = cargar_peft(args.modelo_base, args.adaptador, args.carga)
        nombre_modelo = f"{args.modelo_base} + adaptador"
    else:
        nombre_modelo = f"{args.modelo} (API {args.base_url})"

    resultados = []
    for i, item in enumerate(items, start=1):
        idx = item.get("id", f"item_{i}")
        enunciado = item.get("enunciado", "")
        print(f"[{i}/{len(items)}] {idx} ...", end=" ", flush=True)
        try:
            if args.backend == "api":
                respuesta = responder_api(
                    enunciado, sistema, args.base_url, args.modelo,
                    args.temp, args.max_tokens,
                )
            else:
                respuesta = responder_peft(
                    enunciado, sistema, modelo_peft, tok_peft, args.max_tokens,
                )
        except Exception as e:
            print(f"ERROR: {e}")
            respuesta = ""
            fila = {"id": idx, "eje": item.get("eje", ""), "tipo": item.get("tipo", ""),
                    "clasificacion": "error_inferencia", "estructura_completa": False}
            fila["archivo_respuesta"] = ""
            resultados.append({**clasificar(item, ""), **fila})
            continue

        (salida / "respuestas" / f"{idx}.txt").write_text(respuesta, encoding="utf-8")
        fila = clasificar(item, respuesta)
        fila["archivo_respuesta"] = f"respuestas/{idx}.txt"
        resultados.append(fila)
        print(fila["clasificacion"])

    # Exportar detalle
    columnas = [
        "id", "eje", "tipo", "esperado", "unidad", "num_detectado",
        "todos_los_numeros", "resultado_correcto", "apartados_detectados",
        "apartados_faltantes", "estructura_completa", "contenido_no_solicitado",
        "clasificacion", "archivo_respuesta",
    ]
    escribir_csv(salida / "resultados_detalle.csv", resultados, columnas)

    # Agregado por eje
    resumen = agregar_por_eje(resultados)
    filas_eje = []
    for eje in EJES_SILABO + ["(otros)"]:
        d = resumen[eje]
        if d["n"] == 0:
            continue
        filas_eje.append({
            "eje": eje, "n_total": d["n"], "n_numerico": d["n_num"],
            "n_resultado_correcto": d["n_correcto"],
            "pct_resultado_correcto": pct(d["n_correcto"], d["n_num"]),
            "pct_estructura_completa": pct(d["n_estructura"], d["n"]),
        })
    escribir_csv(
        salida / "resultados_por_eje.csv", filas_eje,
        ["eje", "n_total", "n_numerico", "n_resultado_correcto",
         "pct_resultado_correcto", "pct_estructura_completa"],
    )

    escribir_resumen_md(
        salida / "resumen_resultados.md", resultados, resumen,
        {"modelo": nombre_modelo, "skill": ("(sin skill)" if not sistema else args.skill)},
    )

    # Resumen en consola
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    num = [r for r in resultados if r["tipo"] == "ejercicio_numerico"]
    correctos = [r for r in num if r["clasificacion"] in ("correcto", "correcto_con_formato")]
    manual = [r for r in resultados if r["clasificacion"] == "revision_manual"]
    print(f"Ejercicios evaluados        : {len(resultados)}")
    print(f"Numéricos (auto-calificados) : {len(num)}")
    print(f"Resultado correcto          : {len(correctos)}  ({pct(len(correctos), len(num))})")
    print(f"Requieren rúbrica manual     : {len(manual)}")
    print(f"\nArchivos en: {salida.resolve()}")
    print("  - resultados_detalle.csv")
    print("  - resultados_por_eje.csv")
    print("  - resumen_resultados.md")
    print("  - respuestas/  (respuesta completa por ejercicio)")


if __name__ == "__main__":
    main()
