from graph_builder import construir_grafo_sistema_core

grafo = construir_grafo_sistema_core()
config = {"configurable": {"thread_id": "auditoria_mef_2026"}}

print("=== INICIANDO PROCESAMIENTO ASÍNCRONO DEL GRAFO ===")
estado_inicial = {
    "video_path": "videos/discurso_prueba.srt",
    "rag_retry_count": 0,
    "is_waiting_approval": True
}

for evento in grafo.stream(estado_inicial, config):
    print(f"Nodo ejecutado: {list(evento.keys())}")

# Verificación de detención activa (Human-in-the-Loop)
estado_actual = grafo.get_state(config)
if estado_actual.next:
    print("\n🚨 [HITL ALERTA] El grafo se detuvo mecánicamente antes del nodo:", estado_actual.next)
    print("Estado retenido de forma segura en memoria persistente.")
    
    input("\nPresiona [ENTER] para simular la aprobación manual del auditor y liberar el reporte...")
    
    print("\n=== REANUDANDO GRAFO: PUBLICANDO REPORTE EXPLICABLE ===")
    for evento in grafo.stream(None, config):
        print(f"Nodo ejecutado: {list(evento.keys())}")
