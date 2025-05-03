# app/agent.py

import json
import nltk
from typing_extensions import TypedDict
from typing import List, Dict, Any

# Importa StateGraph y constantes de LangGraph
from langgraph.graph import StateGraph, START, END

# (Ya no se necesita el decorador @tool para las funciones de nodo aquí)
# from langchain_core.tools import tool
# (Ya no se necesita ToolNode aquí)
# from langgraph.prebuilt import ToolNode


# Importa tus *clases* Tool desde app.tool
from app.tool import (
    KaggleIngestionTool,
    PreprocessingTool,
    SentimentToolEn,
    IntentToolEn,
    AlertTool,
    DashboardTool,
)

# Descarga las stopwords UNA sola vez
nltk.download("stopwords", quiet=True)
SW_EN = set(nltk.corpus.stopwords.words("english"))


# Estado que circulará por el grafo
class OmniState(TypedDict, total=False):
    file_path: str                     # Entrada inicial
    docs: List[Dict[str, Any]]         # tras ingest
    cleaned_docs: List[Dict[str, Any]] # tras preprocess
    records: List[Dict[str, Any]]      # tras analyze
    dashboard_path: str                # tras dashboard


# --- Funciones de Nodo ---
# Cada función ahora acepta el estado completo y devuelve un diccionario
# con las claves del estado que desea actualizar.

# app/agent.py

def ingest_node(state: OmniState) -> Dict[str, Any]:
    print("--- Ejecutando Nodo: ingest ---")
    file_path = state.get('file_path')
    # --- Añade esta línea para depurar ---
    print(f"DEBUG: file_path recibido en ingest_node: '{file_path}' (tipo: {type(file_path)})")
    # ------------------------------------
    if not file_path:
        raise ValueError("Falta 'file_path' en el estado inicial.")
    docs_result = KaggleIngestionTool(sample_size=10).run(file_path)
    return {"docs": docs_result}



def preprocess_node(state: OmniState) -> Dict[str, Any]:
    """
    Nodo del grafo: Limpia textos de los documentos en state['docs'].
    Devuelve un diccionario para actualizar la clave 'cleaned_docs' en el estado.
    """
    print("--- Ejecutando Nodo: preprocess ---")
    docs = state.get('docs')
    if docs is None:
        # Es importante manejar el caso donde el estado esperado no existe
        raise ValueError("Falta 'docs' en el estado para preprocess.")

    # Usa la clase Tool correspondiente
    preproc_tool = PreprocessingTool(stopwords=SW_EN)
    cleaned = []
    for doc in docs:
        # Crea una copia para no modificar el estado original directamente en el bucle
        cleaned_doc = doc.copy()
        cleaned_doc["texto"] = preproc_tool.run(doc["texto"])
        cleaned.append(cleaned_doc)
    # Devuelve SOLO la actualización del estado
    return {"cleaned_docs": cleaned}


def analyze_node(state: OmniState) -> Dict[str, Any]:
    """
    Nodo del grafo: Aplica análisis de sentimiento e intención a state['cleaned_docs'].
    Devuelve un diccionario para actualizar la clave 'records' en el estado.
    """
    print("--- Ejecutando Nodo: analyze ---")
    cleaned_docs = state.get('cleaned_docs')
    if cleaned_docs is None:
        raise ValueError("Falta 'cleaned_docs' en el estado para analyze.")

    # Usa las clases Tool correspondientes
    s_tool = SentimentToolEn()
    i_tool = IntentToolEn()
    out = []
    for doc in cleaned_docs:
        s = s_tool.run(doc["texto"])
        i = i_tool.run(doc["texto"])
        out.append({ "id": doc["id"], "canal": doc["canal"], **s, **i })
    # Devuelve SOLO la actualización del estado
    return {"records": out}


def alert_node(state: OmniState) -> Dict[str, Any]: # Devolver Dict vacío es más seguro
    """
    Nodo del grafo: Dispara alertas para registros negativos+queja en state['records'].
    Este nodo es principalmente para efectos secundarios (imprimir alerta).
    No actualiza claves del estado para nodos posteriores, devuelve dict vacío.
    """
    print("--- Ejecutando Nodo: alert ---")
    records = state.get('records')
    if records is None:
        print("Advertencia: No se encontraron 'records' en el estado para alert.")
        return {} # No hay nada que procesar, devuelve dict vacío

    # Usa la clase Tool correspondiente
    alert_tool = AlertTool()
    alerts_triggered_count = 0
    for rec in records:
        # La lógica de filtrado está dentro de alert_tool.run
        result_msg = alert_tool.run(json.dumps(rec, ensure_ascii=False))
        if result_msg != "OK":
            alerts_triggered_count += 1

    print(f"--- Alertas procesadas. Disparadas: {alerts_triggered_count} ---")
    # Este nodo no necesita añadir/modificar el estado para nodos siguientes
    # Podrías devolver {"alerts_triggered": alerts_triggered_count} si fuera útil
    return {}


def dashboard_node(state: OmniState) -> Dict[str, Any]:
    """
    Nodo del grafo: Vuelca state['records'] en un archivo CSV.
    Devuelve un diccionario para actualizar la clave 'dashboard_path' en el estado.
    """
    print("--- Ejecutando Nodo: dashboard ---")
    records = state.get('records')
    if records is None:
        raise ValueError("Falta 'records' en el estado para dashboard.")
     # Asegurarse de que haya registros antes de intentar crear el dashboard
    if not records:
        print("Advertencia: No hay registros ('records') para generar el dashboard.")
        # Puedes decidir si devolver un path vacío/nulo o lanzar error
        return {"dashboard_path": None} # O alguna indicación de que no se generó

    # Usa la clase Tool correspondiente
    dashboard_tool = DashboardTool()
    # Asegúrate de que cada 'record' sea serializable a JSON si usas json.dumps
    # Aquí DashboardTool espera un string con JSONs separados por newline
    batch_json = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    output_path = dashboard_tool.run(batch_json)
    # Devuelve SOLO la actualización del estado
    return {"dashboard_path": output_path}


# --- Construcción del Grafo ---

print("🏗️ Construyendo el grafo...")
builder = StateGraph(OmniState)

# Añade los nodos pasando las funciones directamente (sin ToolNode)
builder.add_node("ingest",     ingest_node)
builder.add_node("preprocess", preprocess_node)
builder.add_node("analyze",    analyze_node)
builder.add_node("alert",      alert_node)
builder.add_node("dashboard",  dashboard_node)

# Define las transiciones entre nodos
builder.add_edge(START,        "ingest")
builder.add_edge("ingest",     "preprocess")
builder.add_edge("preprocess", "analyze")
builder.add_edge("analyze",    "alert")
builder.add_edge("alert",      "dashboard")
builder.add_edge("dashboard",  END)

# Compila el grafo una sola vez
print("✅ Compilando el grafo...")
graph = builder.compile()
print("🏁 Grafo compilado y listo.")


# Ejecución directa desde CLI (para pruebas)
if __name__ == "__main__":
    print("\n--- Ejecutando pipeline desde CLI ---")
    # Asegúrate de que el archivo exista en esta ruta o proporciona la correcta
    input_file = "training.1600000.processed.noemoticon.csv"
    print(f"Archivo de entrada: {input_file}")

    initial_state = {
        "file_path": input_file
    }

    # Invoca el grafo con el estado inicial
    # Usa .stream() si quieres ver los resultados paso a paso o .invoke() para el final
    # result = graph.invoke(initial_state)

    # Usar stream para ver el progreso
    final_result_state = None
    for step in graph.stream(initial_state):
        step_name = list(step.keys())[0]
        step_output = step[step_name]
        print(f"\nOutput del paso '{step_name}':")
        # Imprime una versión abreviada o un resumen del output si es muy grande
        if isinstance(step_output, dict):
             print(json.dumps({k: type(v).__name__ for k,v in step_output.items()}, indent=2))
        else:
            print(step_output)
        final_result_state = step_output # Guarda el último estado completo

    print("\n--- Pipeline completo ---")
    if final_result_state:
        print("Estado final obtenido:")
        # Imprime claves relevantes del estado final
        print(f"  - Documentos procesados (estado intermedio): {'Sí' if 'records' in final_result_state else 'No'}")
        print(f"  - Dashboard generado en: {final_result_state.get('dashboard_path', 'No generado o path no disponible')}")
    else:
         print("No se obtuvo un estado final.")