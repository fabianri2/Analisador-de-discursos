import os
import re
import chromadb
from youtube_transcript_api import YouTubeTranscriptApi

CHROMA_DB_PATH = os.path.join("data_storage", "vector_chroma")
COLLECTION_NAME = "political_discourse_sources"

# 1. Fuentes de YouTube (Solo el ID de 11 caracteres evita cualquier error de URL)
YOUTUBE_SOURCES = [
    {
        "url_or_id": "Z8AiJEiFFx4",
        "espectro": "izquierda",
        "sector": "Reforma Constitucional",
        "fuente": "Cobertura Internacional / El País (YouTube)"
    },
    {
        "url_or_id": "jgtPrArh7Wo",
        "espectro": "derecha",
        "sector": "Seguridad",
        "fuente": "Prensa Nacional / Diario Trome (YouTube)"
    }
]


# 2. Base documental RAG: Benchmarks oficiales + Prensa por Sector
RAG_KNOWLEDGE_BASE = [
    # --- SECTOR 1: REFORMA CONSTITUCIONAL ---
    {
        "text": "Promesa presidencial de adelanto de elecciones generales para 2023 y recorte de mandato ejecutivo y congresal.",
        "tipo": "benchmark_oficial",
        "sector": "Reforma Constitucional",
        "espectro": "oficial",
        "fuente": "JNE / ONPE / Diario Oficial El Peruano",
        "estado_actual": "Incumplido",
        "evidencia_actual": "El Congreso archivó las propuestas de adelanto electoral y el mandato continuó hasta julio de 2026.",
        "hito_1": "30 Ene 2023 | Ejecutivo presenta proyecto de ley de adelanto de elecciones.",
        "hito_2": "Feb-Mar 2023 | Comisión de Constitución del Congreso rechaza y archiva los dictámenes.",
        "hito_3": "Actualidad | Mandato presidencial y congresal sigue vigente sin elecciones adelantadas."
    },
    {
        "text": "El Ejecutivo condicionó el adelanto de elecciones al Congreso y postergó cualquier reforma constitucional para quedarse hasta 2026.",
        "tipo": "prensa_espectro",
        "sector": "Reforma Constitucional",
        "espectro": "izquierda",
        "fuente": "Análisis Crítico / Canal Curwen (YouTube)"
    },
    {
        "text": "La presidenta presentó una iniciativa legislativa con carácter de urgencia para adelantar los comicios dentro del marco constitucional vigente.",
        "tipo": "prensa_espectro",
        "sector": "Reforma Constitucional",
        "espectro": "derecha",
        "fuente": "El Comercio / Editorial Política"
    },

    # --- SECTOR 2: SEGURIDAD ---
    {
        "text": "Compromiso de pacificación nacional, restablecimiento del orden interno y diálogo social sin vulneración de derechos fundamentales.",
        "tipo": "benchmark_oficial",
        "sector": "Seguridad",
        "espectro": "oficial",
        "fuente": "Defensoría del Pueblo / Ministerio del Interior",
        "estado_actual": "Incumplido",
        "evidencia_actual": "Defensoría reportó 49 civiles fallecidos en enfrentamientos y declaratorias prolongadas de estado de emergencia.",
        "hito_1": "Ene 2023 | Anuncio de pacificación y despliegue de fuerzas del orden en corredores viales.",
        "hito_2": "Mar 2023 | Reporte Defensorial confirma 49 fallecidos y cientos de heridos en protestas.",
        "hito_3": "Actualidad | Procesos fiscales abiertos y persistencia de estados de emergencia focalizados."
    },
    {
        "text": "El despliegue policial y militar agravó la conflictividad social en regiones del sur sin abrir mesas reales de diálogo.",
        "tipo": "prensa_espectro",
        "sector": "Seguridad",
        "espectro": "izquierda",
        "fuente": "La República / Cobertura Derechos Humanos"
    },
    {
        "text": "La intervención de la PNP y las Fuerzas Armadas permitió liberar carreteras bloqueadas y restituir el principio de autoridad.",
        "tipo": "prensa_espectro",
        "sector": "Seguridad",
        "espectro": "derecha",
        "fuente": "Juliana Oxenford / Análisis de Coyuntura"
    },

    # --- SECTOR 3: ECONOMÍA ---
    {
        "text": "Plan Con Punche Perú para reactivación económica rápida, control de inflación y crecimiento del PBI regional en 2023.",
        "tipo": "benchmark_oficial",
        "sector": "Economía",
        "espectro": "oficial",
        "fuente": "BCRP / Ministerio de Economía y Finanzas (MEF)",
        "estado_actual": "Parcial",
        "evidencia_actual": "La inflación bajó al rango meta (3.2%), pero el PBI cayó -0.55% entrando en recesión técnica en 2023.",
        "hito_1": "Ene 2023 | Lanzamiento del plan de reactivación económica Con Punche Perú.",
        "hito_2": "Oct 2023 | MEF reconoce oficialmente recesión económica por caída de inversión privada.",
        "hito_3": "Actualidad | Recuperación moderada de indicadores macroeconómicos pero con brecha de empleo."
    },
    {
        "text": "Las cifras del BCRP confirman que el país cayó en recesión económica mientras el discurso oficial prometía reactivación inmediata.",
        "tipo": "prensa_espectro",
        "sector": "Economía",
        "espectro": "izquierda",
        "fuente": "Prensa Independiente / Análisis Económico"
    },
    {
        "text": "La disciplina fiscal del MEF y la autonomía del BCRP lograron frenar la inflación pese a los bloqueos mineros y el fenómeno El Niño.",
        "tipo": "prensa_espectro",
        "sector": "Economía",
        "espectro": "derecha",
        "fuente": "Gestión / Análisis Macroeconómico"
    },

    # --- SECTOR 4: SALUD ---
    {
        "text": "Destrabe de hospitales paralizados, abastecimiento de medicamentos esenciales y cierre de brechas de atención primaria en regiones.",
        "tipo": "benchmark_oficial",
        "sector": "Salud",
        "espectro": "oficial",
        "fuente": "Ministerio de Salud (MINSA) / Contraloría General",
        "estado_actual": "Parcial",
        "evidencia_actual": "Contraloría advirtió desabastecimiento crítico de medicamentos en hospitales regionales y retrasos en obras paralizadas.",
        "hito_1": "Ene 2023 | Promesa de destrabe hospitalario y mejora de atención médica regional.",
        "hito_2": "Ago 2023 | Informes de Contraloría alertan falta de stock de medicinas en establecimientos MINSA.",
        "hito_3": "Actualidad | Avance parcial en expedientes técnicos con obras hospitalarias aún pendientes."
    },
    {
        "text": "Los hospitales regionales continúan con déficit de especialistas y falta de medicamentos básicos pese a los anuncios presupuestales.",
        "tipo": "prensa_espectro",
        "sector": "Salud",
        "espectro": "izquierda",
        "fuente": "Observatorio Ciudadano de Salud"
    },
    {
        "text": "El Ejecutivo transfirió partidas presupuestales extraordinarias para reactivar proyectos hospitalarios estancados de gestiones anteriores.",
        "tipo": "prensa_espectro",
        "sector": "Salud",
        "espectro": "derecha",
        "fuente": "Prensa Nacional / Cobertura Gestión Pública"
    },
        # --- SECTOR 5: EDUCACIÓN ---
    {
        "text": "Rehabilitación de infraestructura educativa en riesgo, entrega oportuna de material escolar y conectividad digital rural.",
        "tipo": "benchmark_oficial",
        "sector": "Educación",
        "espectro": "oficial",
        "fuente": "Ministerio de Educación (MINEDU) / PRONIED",
        "estado_actual": "Incumplido",
        "evidencia_actual": "Más del 50% de locales escolares públicos mantuvieron deficiencias estructurales críticas según reportes de PRONIED.",
        "hito_1": "Ene 2023 | Compromiso de mantenimiento preventivo de colegios antes del inicio de clases.",
        "hito_2": "Mar 2023 | Inicio de año escolar con miles de aulas declaradas inhabitables.",
        "hito_3": "Actualidad | Persiste brecha de infraestructura educativa y conectividad en zonas rurales."
    },
    {
        "text": "La brecha de infraestructura escolar en zonas andinas y amazónicas sigue sin resolverse por baja ejecución presupuestal.",
        "tipo": "prensa_espectro",
        "sector": "Educación",
        "espectro": "izquierda",
        "fuente": "Prensa Regional / Educación Pública"
    },
    {
        "text": "Se garantizó el inicio presencial de clases a nivel nacional y la continuidad de programas de becas universitarias.",
        "tipo": "prensa_espectro",
        "sector": "Educación",
        "espectro": "derecha",
        "fuente": "Editorial Educativa / Prensa Tradicional"
    },

    # --- SECTOR 6: AGRO ---
    {
        "text": "Entrega de fertilizantes, créditos agrarios y proyectos de afianzamiento hídrico para pequeños agricultores afectados por sequías.",
        "tipo": "benchmark_oficial",
        "sector": "Agro",
        "espectro": "oficial",
        "fuente": "Ministerio de Desarrollo Agrario y Riego (MIDAGRI)",
        "estado_actual": "Incumplido",
        "evidencia_actual": "La campaña agrícola sufrió caída de producción por retrasos en bonos agrarios y déficit hídrico en el sur.",
        "hito_1": "Ene 2023 | Anuncio de apoyo financiero e hídrico para la campaña agrícola nacional.",
        "hito_2": "Sep 2023 | Gremios agrarios reportan caída de cosechas y demora en ejecución de reservorios.",
        "hito_3": "Actualidad | Baja ejecución de proyectos de riego tecnificado en regiones altoandinas."
    },
    {
        "text": "Los pequeños productores agrarios denunciaron abandono estatal frente a las heladas, sequías y encarecimiento de insumos.",
        "tipo": "prensa_espectro",
        "sector": "Agro",
        "espectro": "izquierda",
        "fuente": "Prensa Agraria / Regiones del Sur"
    },
    {
        "text": "MIDAGRI impulsó programas de financiamiento rural y seguros catastróficos para mitigar pérdidas climáticas en el campo.",
        "tipo": "prensa_espectro",
        "sector": "Agro",
        "espectro": "derecha",
        "fuente": "Cobertura Económica / Sector Agrario"
    },

    # --- SECTOR 7: MINERÍA E INVERSIONES ---
    {
        "text": "Destrabe de grandes proyectos mineros, paz social en el corredor minero del sur y atracción de inversión privada.",
        "tipo": "benchmark_oficial",
        "sector": "Minería e Inversiones",
        "espectro": "oficial",
        "fuente": "Ministerio de Energía y Minas (MINEM) / BCRP",
        "estado_actual": "Parcial",
        "evidencia_actual": "La inversión minera cayó más de 18% en 2023 por conflictividad social, aunque la producción de cobre creció por Quellaveco.",
        "hito_1": "Ene 2023 | Promesa de destrabe de cartera de proyectos mineros y garantías a la inversión.",
        "hito_2": "Ago 2023 | BCRP reporta contracción de la inversión minera privada en el año fiscal.",
        "hito_3": "Actualidad | Reactivación gradual de exploraciones pero con conflictos latentes en el corredor sur."
    },
    {
        "text": "La paralización del corredor minero y la falta de consulta previa frenaron nuevos proyectos sin resolver demandas ambientales.",
        "tipo": "prensa_espectro",
        "sector": "Minería e Inversiones",
        "espectro": "izquierda",
        "fuente": "Observatorio de Conflictos Mineros"
    },
    {
        "text": "El restablecimiento del tránsito en el corredor minero del sur permitió recuperar exportaciones clave de cobre para la caja fiscal.",
        "tipo": "prensa_espectro",
        "sector": "Minería e Inversiones",
        "espectro": "derecha",
        "fuente": "Gestión / Suplemento Minero"
    },

    # --- SECTOR 8: INFRAESTRUCTURA Y TRANSPORTES ---
    {
        "text": "Ejecución acelerada de obras viales, reconstrucción con cambios en el norte y prevención ante el Fenómeno El Niño.",
        "tipo": "benchmark_oficial",
        "sector": "Infraestructura y Transportes",
        "espectro": "oficial",
        "fuente": "Ministerio de Transportes (MTC) / Contraloría",
        "estado_actual": "Incumplido",
        "evidencia_actual": "Contraloría alertó baja ejecución presupuestal en obras de prevención pluvial y retrasos en carreteras longitudinales.",
        "hito_1": "Ene 2023 | Compromiso de acelerar obras de infraestructura vial y prevención climática.",
        "hito_2": "Jun 2023 | Reportes de Contraloría evidencian menos del 30% de avance en obras de prevención en el norte.",
        "hito_3": "Actualidad | Obras viales emblemáticas siguen en etapa de arbitraje o reanudación parcial."
    },
    {
        "text": "Las regiones del norte volvieron a sufrir inundaciones por la lenta ejecución presupuestal en obras de prevención y drenaje.",
        "tipo": "prensa_espectro",
        "sector": "Infraestructura y Transportes",
        "espectro": "izquierda",
        "fuente": "Prensa Regional Norte / Fiscalización"
    },
    {
        "text": "Se creó la Autoridad Nacional de Infraestructura (ANIN) para destrabar megaproyectos de alto impacto con estándares internacionales.",
        "tipo": "prensa_espectro",
        "sector": "Infraestructura y Transportes",
        "espectro": "derecha",
        "fuente": "El Comercio / Análisis de Infraestructura"
    },

    # --- PERIODO PPK (2016-2018) - EL PERUANO ---
    {
        "text": "Gobierno de Pedro Pablo Kuczynski (PPK): Promesa de Revolución Social, acceso universal a agua y saneamiento al 2021 y reducción del IGV para formalizar la economía.",
        "tipo": "benchmark_oficial",
        "sector": "Economía",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensaje 28 Jul 2016-2017) / MEF / BCRP",
        "estado_actual": "Incumplido",
        "evidencia_actual": "La reducción del IGV fue suspendida por déficit fiscal y la meta de 100% de agua y desagüe quedó inconclusa tras su renuncia en marzo de 2018.",
        "hito_1": "28 Jul 2016 | PPK anuncia Revolución Social, rebaja del IGV y cobertura total de agua potable.",
        "hito_2": "Mar 2018 | Renuncia presidencial en medio de crisis política y baja ejecución de Reconstrucción con Cambios.",
        "hito_3": "Actualidad | Brechas de saneamiento rural y formalización laboral persisten según INEI."
    },
    {
        "text": "Gobierno de PPK: Reconstrucción con Cambios en el norte del país tras el Fenómeno El Niño Costero y destrabe de megaproyectos de transporte.",
        "tipo": "benchmark_oficial",
        "sector": "Infraestructura y Transportes",
        "espectro": "oficial",
        "fuente": "El Peruano / Autoridad para la Reconstrucción con Cambios (ARCC) / Contraloría",
        "estado_actual": "Incumplido",
        "evidencia_actual": "Contraloría reportó ejecución inferior al 20% en obras de reconstrucción vial y colegios durante el periodo 2017-2018.",
        "hito_1": "Jul 2017 | Creación de la Autoridad para la Reconstrucción con Cambios en mensaje presidencial.",
        "hito_2": "2018 | Retrasos en expedientes técnicos y paralización de obras viales en Piura y La Libertad.",
        "hito_3": "Actualidad | Obras de drenaje pluvial fueron transferidas posteriormente a la ANIN."
    },
    {
        "text": "Gobierno de PPK: Cierre de brechas en hospitales públicos, acceso a agua potable y mejora de la calidad educativa.",
        "tipo": "benchmark_oficial",
        "sector": "Salud",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensaje 28 Jul 2016) / MINSA / MVCS",
        "estado_actual": "Parcial",
        "evidencia_actual": "Se destrabaron proyectos urbanos de saneamiento, pero la infraestructura hospitalaria regional mantuvo brechas superiores al 35%.",
        "hito_1": "Jul 2016 | Anuncio de meta de cierre de brechas sanitarias y agua potable.",
        "hito_2": "2017 | Ejecución parcial de inversiones hospitalarias en regiones.",
        "hito_3": "Mar 2018 | Interrupción de gestión por crisis política."
    },

    # --- PERIODO MARTÍN VIZCARRA (2018-2020) - EL PERUANO ---
    {
        "text": "Gobierno de Martín Vizcarra: Reforma política y del sistema de justicia mediante referéndum nacional, bicameralidad y no reelección congresal.",
        "tipo": "benchmark_oficial",
        "sector": "Reforma Constitucional",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensajes 28 Jul 2018-2019) / ONPE / JNE",
        "estado_actual": "Parcial",
        "evidencia_actual": "Se aprobaron 3 de 4 reformas en el referéndum de diciembre 2018 (JNJ, financiamiento partidario y no reelección), pero el Congreso restituyó la bicameralidad y reelección en 2024.",
        "hito_1": "28 Jul 2018 | Convocatoria a referéndum nacional de reforma judicial y política.",
        "hito_2": "Dic 2018 | Ciudadanía aprueba creación de la Junta Nacional de Justicia y no reelección inmediata.",
        "hito_3": "Actualidad | Varias reformas fueron revertidas constitucionalmente por el Congreso en 2024."
    },
    {
        "text": "Gobierno de Martín Vizcarra: Reactivación económica Reactiva Perú, bonos familiares y expansión de camas UCI y hospitales ante la pandemia COVID-19.",
        "tipo": "benchmark_oficial",
        "sector": "Salud",
        "espectro": "oficial",
        "fuente": "El Peruano (Mensaje 28 Jul 2020) / MINSA / MEF",
        "estado_actual": "Parcial",
        "evidencia_actual": "Se ampliaron camas UCI temporales y créditos Reactiva Perú, pero Perú registró la tasa de mortalidad per cápita más alta del mundo según cifras oficiales.",
        "hito_1": "Jul 2020 | Anuncio de inversión histórica en salud y programa Reactiva Perú en mensaje presidencial.",
        "hito_2": "Nov 2020 | Vacancia presidencial interrumpe continuidad de gestión sanitaria.",
        "hito_3": "Actualidad | Déficit estructural de plantas de oxígeno y hospitales de nivel III persiste."
    },
    {
        "text": "Gobierno de Martín Vizcarra: Plan Nacional de Infraestructura para la Competitividad, programa Reactiva Perú y sostenibilidad fiscal.",
        "tipo": "benchmark_oficial",
        "sector": "Economía",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensajes 28 Jul 2019 / 2020) / MEF / BCRP",
        "estado_actual": "Parcial",
        "evidencia_actual": "Reactiva Perú sostuvo la liquidez empresarial en 2020, aunque el PBI cayó 11% por el impacto de la pandemia y la paralización de proyectos.",
        "hito_1": "Jul 2019 | Lanzamiento del Plan Nacional de Infraestructura con 52 proyectos priorizados.",
        "hito_2": "Abr 2020 | Implementación de créditos garantizados Reactiva Perú por S/ 60,000 millones.",
        "hito_3": "Nov 2020 | Inicio de reactivación económica por fases."
    },

    # --- PERIODO FRANCISCO SAGASTI (2020-2021) - EL PERUANO ---
    {
        "text": "Gobierno de transición de Francisco Sagasti: Negociación y compra de vacunas contra la COVID-19 para toda la población adulta y elecciones generales limpias en 2021.",
        "tipo": "benchmark_oficial",
        "sector": "Salud",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Discurso Nov 2020 / Mensaje Jul 2021) / MINSA / JNE",
        "estado_actual": "Cumplido",
        "evidencia_actual": "Se contrataron más de 78 millones de dosis de vacunas (Sinopharm, Pfizer, AstraZeneca) y se realizó la transferencia democrática en julio de 2021.",
        "hito_1": "Nov 2020 | Compromiso de asegurar contratos de vacunas y neutralidad electoral.",
        "hito_2": "Feb-Jul 2021 | Llegada de lotes de vacunas e inicio de vacunación masiva nacional.",
        "hito_3": "28 Jul 2021 | Entrega de mando presidencial tras elecciones generales certificadas por JNE."
    },
    {
        "text": "Gobierno de transición de Francisco Sagasti: Fortalecimiento institucional de la Policía Nacional, apoyo de las Fuerzas Armadas en la emergencia y respeto irrestricto a los derechos humanos.",
        "tipo": "benchmark_oficial",
        "sector": "Seguridad",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Discurso de Investidura 17 Nov 2020) / MININTER / Defensoría del Pueblo",
        "estado_actual": "Cumplido",
        "evidencia_actual": "Se ejecutó la renovación de mandos policiales y se garantizó el orden interno pacífico durante las jornadas electorales de 2021 sin vulneraciones graves reportadas por Defensoría.",
        "hito_1": "17 Nov 2020 | Discurso de asunción comprometiendo reforma policial y respeto a derechos humanos.",
        "hito_2": "Abr-Jun 2021 | Despliegue conjunto de PNP y Fuerzas Armadas garantizando seguridad en primera y segunda vuelta.",
        "hito_3": "28 Jul 2021 | Cierre de gestión de transición en calma social e institucional."
    },
    {
        "text": "Gobierno de transición de Francisco Sagasti: Defensa del orden constitucional, Bicentenario de la Independencia y garantía de elecciones limpias y transparentes en 2021.",
        "tipo": "benchmark_oficial",
        "sector": "Reforma Constitucional",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensajes Nov 2020 / Jul 2021) / JNE / ONPE",
        "estado_actual": "Cumplido",
        "evidencia_actual": "El Ejecutivo mantuvo estricta neutralidad electoral reconocida por misiones internacionales (OEA y Unión Europea) y culminó con la proclamación oficial y cambio de mando el 28 de julio de 2021.",
        "hito_1": "Nov 2020 | Compromiso ante el Congreso de asegurar elecciones limpias en el Bicentenario.",
        "hito_2": "Jun 2021 | Respaldo a la autonomía de los organismos electorales (JNE y ONPE).",
        "hito_3": "28 Jul 2021 | Entrega de la banda presidencial tras proclamación constitucional."
    },
    {
        "text": "Gobierno de transición de Francisco Sagasti: Reactivación económica responsable, apoyo financiero a las MYPES y protección social frente a la segunda ola de la pandemia.",
        "tipo": "benchmark_oficial",
        "sector": "Economía",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensaje 28 Jul 2021) / MEF / BCRP",
        "estado_actual": "Parcial",
        "evidencia_actual": "El PBI rebote creció 13.3% en 2021 impulsado por exportaciones y reapertura gradual, aunque la recuperación del empleo formal y las microempresas fue desigual.",
        "hito_1": "Nov 2020 | Anuncio de equilibrio entre control sanitario y reactivación económica.",
        "hito_2": "Mar 2021 | Ampliación de programas de apoyo crediticio FAE-Mype y bonos focalizados.",
        "hito_3": "Jul 2021 | Cierre de transición con reservas internacionales estables y recuperación macroeconómica en curso."
    },


    # --- PERIODO PEDRO CASTILLO (2021-2022) - EL PERUANO ---
    {
        "text": "Gobierno de Pedro Castillo: Convocatoria a Asamblea Constituyente para redactar una nueva Constitución Política y masificación del gas de Camisea.",
        "tipo": "benchmark_oficial",
        "sector": "Reforma Constitucional",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensaje 28 Jul 2021) / Congreso de la República / MINEM",
        "estado_actual": "Incumplido",
        "evidencia_actual": "El proyecto de ley de Asamblea Constituyente fue archivado en la Comisión de Constitución del Congreso en mayo de 2022 y la masificación del gas mantuvo cobertura limitada.",
        "hito_1": "28 Jul 2021 | Anuncio en Mensaje a la Nación de impulsar una Asamblea Constituyente.",
        "hito_2": "May 2022 | Archivo definitivo de la iniciativa legislativa en el Congreso.",
        "hito_3": "Dic 2022 | Destitución presidencial tras intento de ruptura del orden constitucional."
    },
    {
        "text": "Gobierno de Pedro Castillo: Segunda Reforma Agraria, industrialización del agro y compra internacional de fertilizantes (urea) para pequeños agricultores.",
        "tipo": "benchmark_oficial",
        "sector": "Agro",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensaje 28 Jul 2022) / MIDAGRI / Contraloría",
        "estado_actual": "Incumplido",
        "evidencia_actual": "Cuatro licitaciones internacionales para compra de urea fueron anuladas por observaciones de Contraloría, sin concretarse la entrega oportuna en la campaña agrícola 2022.",
        "hito_1": "Oct 2021 | Lanzamiento simbólico de la Segunda Reforma Agraria en Cusco.",
        "hito_2": "Jul 2022 | Promesa de llegada inmediata de fertilizantes para campaña agrícola.",
        "hito_3": "Dic 2022 | Cierre de gestión sin adquisición de urea y reemplazo tardío por Fertiabono."
    },
    {
        "text": "Gobierno de Pedro Castillo: Declaratoria de emergencia en educación, ingreso libre a universidades públicas y aumento presupuestal del sector educativo.",
        "tipo": "benchmark_oficial",
        "sector": "Educación",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensaje 28 Jul 2021) / MINEDU",
        "estado_actual": "Incumplido",
        "evidencia_actual": "El proyecto de ingreso libre a universidades públicas no llegó a implementarse ni obtuvo viabilidad presupuestal en el Congreso.",
        "hito_1": "Jul 2021 | Promesa de declarar en emergencia la educación e implementar ingreso libre universitario.",
        "hito_2": "2022 | Retorno presencial a clases tras pandemia con brechas de infraestructura escolar.",
        "hito_3": "Dic 2022 | Proyecto de ingreso libre queda sin ejecución."
    },
    {
        "text": "Gobierno de Pedro Castillo: Estabilidad macroeconómica, apoyo económico mediante bonos Yanapay y lucha contra la inseguridad ciudadana.",
        "tipo": "benchmark_oficial",
        "sector": "Economía",
        "espectro": "oficial",
        "fuente": "Diario Oficial El Peruano (Mensajes 28 Jul 2021 / 2022) / MEF / BCRP",
        "estado_actual": "Parcial",
        "evidencia_actual": "La autonomía del BCRP preservó la estabilidad monetaria, pero la alta rotación ministerial frenó la inversión privada en 2022.",
        "hito_1": "Jul 2021 | Ratificación de respeto a la estabilidad macroeconómica y entrega de bono Yanapay.",
        "hito_2": "2022 | Desaceleración de la inversión privada por inestabilidad política.",
        "hito_3": "Dic 2022 | Cierre de periodo tras crisis institucional."
    },
]

def _extract_video_id(url_or_id: str) -> str:
    clean = url_or_id.strip()
    if len(clean) == 11 and "/" not in clean and "." not in clean:
        return clean
    match = re.search(r"(?:v=|youtu\.be\/)([0-9A-Za-z_-]{11})", clean)
    return match.group(1) if match else ""



def _fetch_youtube_chunks(video_id: str, chunk_size: int = 450) -> list[str]:
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=["es", "es-419", "es-PE"])
        full_text = " ".join(item.text.replace("\n", " ") for item in transcript_list)
        words = full_text.split()
        chunks, current_chunk, current_len = [], [], 0
        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [], 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    except Exception as exc:
        print(f"  ⚠️ Subtítulos no disponibles para '{video_id}' ({exc}). Usando base RAG curada.")
        return []


def build_vector_database():
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents, metadatas, ids = [], [], []

    for idx, item in enumerate(RAG_KNOWLEDGE_BASE, start=1):
        meta = {
            "tipo": item["tipo"],
            "sector": item["sector"],
            "espectro": item["espectro"],
            "fuente": item["fuente"],
            "estado_actual": item.get("estado_actual", "No aplica"),
            "evidencia_actual": item.get("evidencia_actual", ""),
            "hito_1": item.get("hito_1", ""),
            "hito_2": item.get("hito_2", ""),
            "hito_3": item.get("hito_3", "")
        }
        documents.append(item["text"])
        metadatas.append(meta)
        ids.append(f"rag_{item['sector'].lower().replace(' ', '_')}_{idx}")

    yt_counter = 1
    for source in YOUTUBE_SOURCES:
        vid = _extract_video_id(source["url_or_id"])
        chunks = _fetch_youtube_chunks(vid)
        for chunk in chunks[:4]:
            documents.append(chunk)
            metadatas.append({
                "tipo": "prensa_espectro",
                "sector": source["sector"],
                "espectro": source["espectro"],
                "fuente": source["fuente"],
                "estado_actual": "No aplica",
                "evidencia_actual": "",
                "hito_1": "",
                "hito_2": "",
                "hito_3": ""
            })
            ids.append(f"yt_{vid}_{yt_counter}")
            yt_counter += 1

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Base RAG ChromaDB reconstruida con {collection.count()} documentos sectoriales en '{CHROMA_DB_PATH}'.")


if __name__ == "__main__":
    build_vector_database()

