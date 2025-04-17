import argparse
import logging
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI, APIError, APITimeoutError, BadRequestError

# Import schemas properly from schemas.py and json_schemas.py
from schemas import argentinian_names, illneses, SPANISH_CHARS, PatientScript
from json_schemas import patient_script_json_schema

# --- Configure logging ---

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds
CONTEXT_LENGTH_ERROR_MARKER = "CONTEXT_LENGTH_EXCEEDED" # Sentinel value

# --- Define JSON Schemas for API Calls ---

# Schema for the initial setup response (profile + first message)
setup_response_schema = {
    "type": "object",
    "properties": {
        "patient_profile": patient_script_json_schema,  # Use the structured patient script schema
        "first_message": {"type": "string", "description": "The first message from the doctor to the patient to start the conversation. In Spanish"}
    },
    "required": ["patient_profile", "first_message"]
}

# Schema for the doctor's response during dialogue simulation
doctor_response_schema = {
    "type": "object",
    "properties": {
        "doctor_response": {"type": "string", "description": "The doctor's next single line of dialogue only."}
    },
    "required": ["doctor_response"]
}

# Schema for the final evaluation scores
evaluation_scores_schema = {
    "type": "object",
    "properties": {
        "consistency_score": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Score (1-5) for patient consistency."},
        "realism_score": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Score (1-5) for patient realism."},
        "adherence_score": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Score (1-5) for patient adherence to scenario."},
        "justification": {"type": "string", "description": "Brief justification for the scores provided."}
    },
    "required": ["consistency_score", "realism_score", "adherence_score", "justification"]
}

def call_openai_api(client, model_id, messages, temperature=0.7, max_tokens=500, schema=None):
    """Helper function to call OpenAI API with retries and optional JSON schema enforcement."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response_format = {"type": "json_object", "schema": schema} if schema else None

            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format # Use the potentially schema-defined format
            )
            return completion.choices[0].message.content
        except BadRequestError as e:
            # Check if it's the context length error
            if "maximum context length" in str(e).lower():
                logger.warning(f"Context length exceeded for model {model_id}. Request: {len(messages)} messages. Error: {e}")
                return CONTEXT_LENGTH_ERROR_MARKER # Return sentinel value immediately
            else:
                # Handle other Bad Request errors (e.g., invalid JSON prompt) - maybe don't retry these?
                logger.error(f"BadRequestError (non-context): {e}", exc_info=True)
                return None # Or raise e depending on desired handling
        except (APIError, APITimeoutError) as e:
            retries += 1
            logger.warning(f"API Error: {e}. Retrying ({retries}/{MAX_RETRIES}) in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}", exc_info=True)
            return None # Treat unexpected errors as failures after logging
    logger.error(f"Failed to get response from {client.base_url} after {MAX_RETRIES} retries.")
    return None

# --- Updated Evaluation Function ---

def run_evaluation(evaluator_client, patient_client, evaluator_model, patient_model, scenario, max_turns):
    """Runs a single evaluation dialogue and scoring."""
    # Ensure scenario unpacking matches the structure from schemas.py (name, [illness_name, illness_desc], age)
    name, illness_data, age = scenario
    illness_name, illness_desc = illness_data # Unpack the inner list/tuple

    logger.info(f"Starting evaluation for: {name}, {age}, {illness_name}")

    # Phase 1: Evaluator generates profile and first doctor message
    # Use a structured approach similar to utils.py for patient script generation
    setup_prompt = f"""Eres un evaluador experto simulando ser un médico para interactuar con una IA paciente. Tu objetivo es evaluar la consistencia, realismo y adherencia de la IA paciente a su perfil asignado.

Primero, crea un perfil detallado en forma de ficha médica para la IA paciente basado en estos datos:
Nombre: {name}
Edad: {age}
Condición Principal: {illness_name} - {illness_desc}

El perfil debe seguir una estructura específica que incluya:
- Demografía (edad, sexo, ocupación, nivel educativo)
- Historia médica (condiciones, medicamentos, alergias, historial quirúrgico)
- Síntomas actuales (queja principal, duración, severidad, síntomas asociados)
- Detalles personales (hábitos, dinámica familiar, trabajo, historia social, salud mental)
- Factores conductuales y cognitivos
- Utilización de servicios de salud

Este perfil estructurado se usará como prompt de sistema para la IA paciente.

Luego, inicia la conversación con una pregunta o comentario inicial apropiado como si fueras el médico dirigiéndote al paciente por primera vez.

Responde únicamente en formato JSON adhiriéndose estrictamente al siguiente esquema:
{json.dumps(setup_response_schema)}
"""
    logger.debug("Requesting setup from evaluator...")
    setup_response_str = call_openai_api(
        evaluator_client,
        evaluator_model,
        [{"role": "system", "content": setup_prompt}],
        temperature=0.5,
        max_tokens=1000,  # Increased to accommodate the larger structured profile
        schema=setup_response_schema
    )

    if not setup_response_str:
        logger.error("Failed to get setup response from evaluator.")
        return None

    try:
        setup_data = json.loads(setup_response_str)
        patient_profile = setup_data['patient_profile']  # This is now a structured object
        patient_profile_str = json.dumps(patient_profile, ensure_ascii=False, indent=2)  # For display in logs
        first_doctor_message = setup_data['first_message']
        logger.debug(f"Evaluator generated structured profile: {patient_profile_str[:200]}...")  # Log first part
        logger.debug(f"Evaluator first message: {first_doctor_message}")
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse setup response from evaluator: {e}\nResponse: {setup_response_str}")
        return None

    # Phase a: Format patient profile for system prompt
    # Convert the structured profile to a format suitable for the patient system prompt
    try:
        # Format the patient profile for the system prompt in a conversational way
        formatted_profile = f"""Eres un paciente simulado con el siguiente perfil médico:
Nombre: {patient_profile.get('Name', name)}
Demografía: {age} años, {patient_profile.get('Demographics', {}).get('sex', 'no especificado')}
Ocupación: {patient_profile.get('Demographics', {}).get('occupation', 'no especificado')}
Síntoma principal: {patient_profile.get('Current_Symptoms', {}).get('chief_complaint', illness_name)}
Condiciones médicas: {', '.join(patient_profile.get('Medical_History', {}).get('conditions', [illness_name]))}

Responde como este paciente de manera realista y consistente con toda esta información."""
    except AttributeError:
        # If patient_profile is not properly structured, use a fallback approach
        logger.warning("Patient profile not properly structured, using fallback format")
        formatted_profile = f"""Eres un paciente simulado. 
Nombre: {name}
Edad: {age}
Condición Principal: {illness_name} - {illness_desc}

Responde de manera realista y consistente con este perfil."""

    # Phase 2: Dialogue Simulation
    patient_messages = [
        {"role": "system", "content": formatted_profile},
        {"role": "user", "content": first_doctor_message} # In patient context, doctor is 'user'
    ]
    
    # For evaluator, we'll keep the full structured profile
    full_transcript_for_eval = [
        {"role": "system", "content": f"Perfil del Paciente Asignado: {patient_profile_str}"},
        {"role": "user", "content": first_doctor_message} # In evaluator context, doctor is 'user'
    ]

    logger.info("Starting dialogue simulation...")
    dialogue_truncated = False # Flag to indicate if context limit was hit
    for turn in range(max_turns):
        logger.debug(f"Turn {turn + 1}/{max_turns}")

        # Patient Turn
        logger.debug("Getting patient response...")
        patient_response = call_openai_api(
            patient_client,
            patient_model,
            patient_messages,
            temperature=0.8, # Allow more creativity for patient
            max_tokens=250 # This contributes to context length
        )

        if patient_response == CONTEXT_LENGTH_ERROR_MARKER:
            logger.warning(f"Context length exceeded during patient turn {turn + 1}. Truncating dialogue.")
            dialogue_truncated = True
            break # Exit dialogue loop

        if not patient_response:
            logger.warning("Failed to get response from patient model. Ending dialogue.")
            break

        patient_messages.append({"role": "assistant", "content": patient_response})
        full_transcript_for_eval.append({"role": "assistant", "content": patient_response}) # Patient is 'assistant'
        logger.debug(f"Patient: {patient_response}")

        # Check if conversation should end (optional basic check)
        if len(patient_response.split()) < 5 and turn > 3: # Arbitrary short response check
             logger.info("Patient response seems short, potentially ending conversation.")
             # break # Decide if short responses should end the convo

        # Doctor (Evaluator) Turn
        evaluator_prompt_content = f"""Continúa la conversación actuando como el médico. Mantén la interacción enfocada en evaluar la condición del paciente según su perfil: '{patient_profile_str}'. Haz preguntas relevantes o comentarios apropiados para una consulta médica. Sé conciso.

Historial hasta ahora (últimos intercambios):
{json.dumps(full_transcript_for_eval[-4:], ensure_ascii=False, indent=2)}

Proporciona únicamente tu siguiente línea de diálogo como médico. Responde en formato JSON adhiriéndose estrictamente al siguiente esquema:
{json.dumps(doctor_response_schema)}
"""

        evaluator_turn_messages = full_transcript_for_eval + [{"role": "system", "content": evaluator_prompt_content}]

        logger.debug("Getting doctor (evaluator) response...")
        doctor_response_json_str = call_openai_api(
            evaluator_client,
            evaluator_model,
            evaluator_turn_messages,
            temperature=0.6,
            max_tokens=200, # This also contributes to context length
            schema=doctor_response_schema # Pass the schema here
        )

        if doctor_response_json_str == CONTEXT_LENGTH_ERROR_MARKER:
            logger.warning(f"Context length exceeded during doctor (evaluator) turn {turn + 1}. Truncating dialogue.")
            dialogue_truncated = True
            break # Exit dialogue loop

        if not doctor_response_json_str:
            logger.warning("Failed to get response from evaluator model (as doctor). Ending dialogue.")
            break

        try:
            # Parse the JSON and extract the doctor's dialogue
            doctor_response_data = json.loads(doctor_response_json_str)
            doctor_response = doctor_response_data['doctor_response']
            if not isinstance(doctor_response, str) or not doctor_response.strip():
                 raise ValueError("Invalid 'doctor_response' content")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse doctor response JSON from evaluator: {e}\nRaw Response: {doctor_response_json_str}")
            # Decide how to handle - skip turn, end dialogue, or use raw string? Let's end dialogue for now.
            break

        # Add doctor response to both histories
        patient_messages.append({"role": "user", "content": doctor_response})
        full_transcript_for_eval.append({"role": "user", "content": doctor_response})
        logger.debug(f"Doctor (Eval): {doctor_response}")

        # Optional: Add more sophisticated end-of-conversation detection here

    if dialogue_truncated:
        logger.info("Dialogue simulation finished early due to context length limit.")
    else:
        logger.info("Dialogue simulation finished.")

    # Phase 3: Evaluation Scoring
    scoring_prompt = f"""Basado en la siguiente transcripción de la conversación (que puede haber sido truncada por límites de contexto) y el perfil inicial del paciente, evalúa el desempeño de la IA paciente. Proporciona puntajes del 1 al 5 (1=Muy Malo, 5=Excelente) para:
1.  **Consistencia:** ¿Mantuvo el paciente una persona y historial médico consistentes durante la conversación?
2.  **Realismo:** ¿Fue el diálogo del paciente creíble y médicamente plausible?
3.  **Adherencia:** ¿Se apegó el paciente al escenario y condición asignados?

Formatea tu respuesta únicamente como un objeto JSON adhiriéndose estrictamente al siguiente esquema:
{json.dumps(evaluation_scores_schema)}

**Perfil del Paciente Asignado:**
{patient_profile_str}

**Transcripción (puede estar incompleta):**
{json.dumps(full_transcript_for_eval, ensure_ascii=False, indent=2)}

**Tu evaluación en formato JSON:**
"""
    logger.debug("Requesting evaluation scores...")
    evaluation_response_str = call_openai_api(
        evaluator_client,
        evaluator_model,
        [{"role": "system", "content": scoring_prompt}],
        temperature=0.2, # Low temp for consistent scoring format
        max_tokens=300, # Ensure scoring prompt itself doesn't exceed limits
        schema=evaluation_scores_schema # Pass the schema here
    )

    if not evaluation_response_str:
        logger.error("Failed to get evaluation scores from evaluator.")
        scores = {"error": "Failed to get evaluation response"}
    else:
        try:
            scores = json.loads(evaluation_response_str)
            # Basic validation
            for key in ['consistency_score', 'realism_score', 'adherence_score']:
                if not isinstance(scores.get(key), int) or not (1 <= scores[key] <= 5):
                     raise ValueError(f"Invalid score for {key}: {scores.get(key)}")
            if not isinstance(scores.get('justification'), str):
                raise ValueError("Invalid justification")
            logger.info(f"Evaluation scores received: {scores}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse evaluation scores: {e}\nResponse: {evaluation_response_str}")
            scores = {"error": f"Failed to parse evaluation response: {e}", "raw_response": evaluation_response_str}

    # Log Results
    result = {
        "scenario": {
            "name": name,
            "age": age,
            "illness_name": illness_name,
            "illness_desc": illness_desc,
            "generated_profile": patient_profile  # Store the structured profile
        },
        "transcript": full_transcript_for_eval,
        "evaluation": scores
    }
    # Add truncation info to the result
    result["dialogue_truncated"] = dialogue_truncated
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a patient simulation LLM using an evaluator LLM.")
    parser.add_argument("--evaluator_url", required=True, help="Base URL for the Evaluator LLM API")
    parser.add_argument("--evaluator_api_key", required=True, help="API Key for the Evaluator LLM")
    parser.add_argument("--patient_url", required=True, help="Base URL for the Patient LLM API")
    parser.add_argument("--patient_api_key", default="no_key", help="API Key for the Patient LLM (optional)")
    parser.add_argument("--n_evals", type=int, default=10, help="Number of evaluation scenarios to run")
    parser.add_argument("--output_file", type=str, default='evaluation_results.jsonl', help="Path to JSONL output file")
    parser.add_argument("--max_turns", type=int, default=15, help="Maximum number of dialogue turns (1 turn = 1 patient + 1 doctor response)") # Defaulting to 15 (30 total messages)
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Set the logging level")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    logger.debug(f"Set logging level to: {args.log_level}")

    # Initialize OpenAI Clients
    logger.info(f"Initializing Evaluator client for: {args.evaluator_url}")
    evaluator_client = OpenAI(base_url=args.evaluator_url, api_key=args.evaluator_api_key, timeout=60.0)
    logger.info(f"Initializing Patient client for: {args.patient_url}")
    patient_client = OpenAI(base_url=args.patient_url, api_key=args.patient_api_key, timeout=60.0)

    # Get available models (optional, good for verification)
    try:
        evaluator_models = evaluator_client.models.list()
        evaluator_model_id = evaluator_models.data[0].id
        logger.info(f"Using Evaluator model: {evaluator_model_id}")
    except Exception as e:
        logger.warning(f"Could not automatically determine evaluator model: {e}. Ensure the endpoint serves a model.")
        evaluator_model_id = "evaluator-model" # Placeholder

    try:
        patient_models = patient_client.models.list()
        patient_model_id = patient_models.data[0].id
        logger.info(f"Using Patient model: {patient_model_id}")
    except Exception as e:
        logger.warning(f"Could not automatically determine patient model: {e}. Ensure the endpoint serves a model.")
        patient_model_id = "patient-model" # Placeholder


    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_path}")

    # Generate scenarios
    # Using the same approach as in synth_dialogues.py
    scenarios = [(random.choice(argentinian_names),
                  random.choice(illneses), # This should now correctly be [name, desc]
                  random.randint(18, 71))
                 for _ in range(args.n_evals)]
    logger.info(f"Generated {len(scenarios)} scenarios for evaluation.")

    # Run evaluations
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i, scenario in enumerate(tqdm(scenarios, desc="Running Evaluations")):
            logger.info(f"--- Starting Evaluation {i+1}/{args.n_evals} ---")
            result = run_evaluation(
                evaluator_client,
                patient_client,
                evaluator_model_id,
                patient_model_id,
                scenario,
                args.max_turns
            )
            if result:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                logger.info(f"--- Finished Evaluation {i+1}/{args.n_evals} ---")
            else:
                logger.error(f"Evaluation {i+1} failed for scenario: {scenario}")
                # Optionally write a failure marker to the output
                f_out.write(json.dumps({"error": "Evaluation failed", "scenario": scenario}) + '\n')

    logger.info("Evaluation process completed.")

