# NOTA: Agente encargado del análisis sintáctico forense y opacidad del texto.

def text_opacity_analyst_agent(state: dict) -> dict:
    """
    Analyzes the syntax complexity and opacity of the presidential speech text.
    """
    # Recuperamos el texto del estado compartido de LangGraph
    speech_text = state.get("transcribed_text", "")
    
    # Lógica computacional básica para el MVP del Sprint 1
    words_list = speech_text.split()
    total_words = len(words_list)
    
    # Cálculo del índice de opacidad base
    opacity_score = 0.0
    if total_words > 0:
        opacity_score = 65.5  # Simulación de métrica Fog Index / opacidad léxica
        
    print(f"[Syntactic Agent] Speech opacity calculated: {opacity_score}%")
    
    # Retorna la actualización al estado en formato estructurado
    return {"syntactic_metrics": {"opacity_score": opacity_score}}