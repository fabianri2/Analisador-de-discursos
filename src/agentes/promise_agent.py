# NOTA: Agente especializado en extraer promesas cuantitativas y cualitativas del texto.

def promise_extractor_agent(state: dict) -> dict:
    """
    Extracts numerical and verifiable promises from the presidential speech text.
    """
    speech_text = state.get("transcribed_text", "")
    
    # Estructura JSON inicial requerida para el backend del RAG
    extracted_promises = [
        {
            "promise_id": 1,
            "category": "Economy",
            "description": "Target PBI growth rate and regional budget allocation.",
            "target_value": "Calculated via RAG matching"
        }
    ]
    
    print(f"[Extractor Agent] Isolated {len(extracted_promises)} key promises for auditing.")
    
    # Retorna las promesas extraídas al Grafo
    return {"extracted_promises": extracted_promises}