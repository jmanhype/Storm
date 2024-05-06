import os
from anthropic import Anthropic
import re
from rich.console import Console
from rich.panel import Panel
from datetime import datetime
import json
from tavily import TavilyClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import cognee

# Set up the Anthropic API client
client = Anthropic(api_key="")

# Set up Cognee environment variables
os.environ["WEAVIATE_URL"] = "http://192.168.1.163:8080"
os.environ["WEAVIATE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

# Available Claude models:
# Claude 3 Opus     claude-3-opus-20240229
# Claude 3 Sonnet   claude-3-sonnet-20240229
# Claude 3 Haiku    claude-3-haiku-20240307

ORCHESTRATOR_MODEL = "claude-3-opus-20240229"
SUB_AGENT_MODELS = ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
REFINER_MODEL = "claude-3-opus-20240229"

def calculate_subagent_cost(model, input_tokens, output_tokens):
    # Pricing information per model
    pricing = {
        "claude-3-opus-20240229": {"input_cost_per_mtok": 15.00, "output_cost_per_mtok": 75.00},
        "claude-3-haiku-20240307": {"input_cost_per_mtok": 0.25, "output_cost_per_mtok": 1.25},
        "claude-3-sonnet-20240229": {"input_cost_per_mtok": 3.00, "output_cost_per_mtok": 15.00},
    }

    # Calculate cost
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input_cost_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output_cost_per_mtok"]
    total_cost = input_cost + output_cost

    return total_cost

# Initialize the Rich Console
console = Console()

def enhanced_opus_orchestrator(objective, file_content=None, previous_results=None, use_search=False):
    response_text, file_content, search_query = opus_orchestrator(objective, file_content, previous_results, use_search)
    return response_text, file_content, search_query

def opus_orchestrator(objective, file_content=None, previous_results=None, use_search=False):
    console.print(f"\n[bold]Calling Orchestrator for your objective[/bold]")
    if file_content:
        console.print(Panel(f"File content:\n{file_content}", title="[bold blue]File Content[/bold blue]", title_align="left", border_style="blue"))

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "Please process this research question into sub-questions: " + objective
        }]
    }]

    opus_response = client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=4096,
        messages=messages
    )
    response_text = opus_response.content[0].text
    console.print(f"Input Tokens: {opus_response.usage.input_tokens}, Output Tokens: {opus_response.usage.output_tokens}")

    # Directly log and return the text response without attempting JSON parsing
    console.print(f"Raw response text: {response_text}")

    return response_text, file_content, None  # No search query to return since no JSON parsing is doneS

def handle_search_queries(query):
    search_results = cognee.search("SIMILARITY", query)
    return search_results

import asyncio

async def enhanced_haiku_sub_agent(prompt, search_query=None, model=SUB_AGENT_MODELS[0], use_search=False, max_tokens=1500):
    await cognee.add(prompt)  # Add the prompt to the knowledge base
    await cognee.cognify()  # Process the added information

    context_response = None
    qna_response = None

    if search_query and use_search:
        # Perform a QnA search based on the search query
        qna_response = handle_search_queries(search_query)
        console.print(f"QnA Response: {qna_response}", style="yellow")

        # Retrieve search context limited by token count
        context_response = "\n".join(handle_search_queries(search_query))
        console.print(f"Context Response: {context_response}", style="yellow")

    # Prepare the messages array with only the prompt initially
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    # Add search results to the messages if there are any
    if qna_response:
        messages[0]["content"].append({"type": "text", "text": f"\nQnA Search Results:\n{qna_response}"})
    if context_response:
        messages[0]["content"].append({"type": "text", "text": f"\nSearch Context:\n{context_response}"})

    haiku_response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=messages
    )

    response_text = haiku_response.content[0].text
    console.print(f"Input Tokens: {haiku_response.usage.input_tokens}, Output Tokens: {haiku_response.usage.output_tokens}")
    total_cost = calculate_subagent_cost(model, haiku_response.usage.input_tokens, haiku_response.usage.output_tokens)
    console.print(f"Sub-agent Cost: ${total_cost:.4f}")

    console.print(Panel(response_text, title=f"[bold blue]Sub-agent Result[/bold blue]", title_align="left", border_style="blue", subtitle=f"Task completed by {model}"))

    return response_text

def opus_refine(objective, sub_task_results, filename, projectname):
    console.print("\nCalling Opus to provide the refined final output for your objective:")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Objective: " + objective + "\n\nSub-task results:\n" + "\n".join(sub_task_results) + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include the necessary import statements at the beginning of the file. Then, include ONLY the file name NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\n​python\n<code>\n​\n4. Unit Test Files: Please also create unit test files for each code file. Use the naming convention 'test_<filename>.py' for the unit test files. Include the necessary import statements and write unit tests for the critical functionalities of each code file. Follow the same format as the code files, with 'Filename: test_<filename>.py' followed by the code block."}
            ]
        }
    ]

    opus_response = client.messages.create(
        model=REFINER_MODEL,
        max_tokens=4096,
        messages=messages
    )

    response_text = opus_response.content[0].text.strip()
    console.print(f"Input Tokens: {opus_response.usage.input_tokens}, Output Tokens: {opus_response.usage.output_tokens}")
    total_cost = calculate_subagent_cost(REFINER_MODEL, opus_response.usage.input_tokens, opus_response.usage.output_tokens)
    console.print(f"Refine Cost: ${total_cost:.4f}")

    console.print(Panel(response_text, title="[bold green]Final Output[/bold green]", title_align="left", border_style="green"))
    return response_text

def create_folder_structure(project_name, folder_structure, code_blocks):
    # Create the project folder
    try:
        os.makedirs(project_name, exist_ok=True)
        console.print(Panel(f"Created project folder: [bold]{project_name}[/bold]", title="[bold green]Project Folder[/bold green]", title_align="left", border_style="green"))
    except OSError as e:
        console.print(Panel(f"Error creating project folder: [bold]{project_name}[/bold]\nError: {e}", title="[bold red]Project Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        return

    # Recursively create the folder structure and files
    create_folders_and_files(project_name, folder_structure, code_blocks)

def create_folders_and_files(current_path, structure, code_blocks):
    for key, value in structure.items():
        path = os.path.join(current_path, key)
        if isinstance(value, dict):
            try:
                os.makedirs(path, exist_ok=True)
                console.print(Panel(f"Created folder: [bold]{path}[/bold]", title="[bold blue]Folder Creation[/bold blue]", title_align="left", border_style="blue"))
                create_folders_and_files(path, value, code_blocks)
            except OSError as e:
                console.print(Panel(f"Error creating folder: [bold]{path}[/bold]\nError: {e}", title="[bold red]Folder Creation Error[/bold red]", title_align="left", border_style="red"))
        else:
            code_content = next((code for file, code in code_blocks if file == key), None)
            if code_content:
                try:
                    with open(path, 'w') as file:
                        file.write(code_content)
                    console.print(Panel(f"Created file: [bold]{path}[/bold]", title="[bold green]File Creation[/bold green]", title_align="left", border_style="green"))
                except IOError as e:
                    console.print(Panel(f"Error creating file: [bold]{path}[/bold]\nError: {e}", title="[bold red]File Creation Error[/bold red]", title_align="left", border_style="red"))
            else:
                console.print(Panel(f"Code content not found for file: [bold]{key}[/bold]", title="[bold yellow]Missing Code Content[/bold yellow]", title_align="left", border_style="yellow"))

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# Get the objective from user input
objective = input("Please enter your objective with or without a text file path: ")

# Check if the input contains a file path
if "./" in objective or "/" in objective:
    # Extract the file path from the objective
    file_path = re.findall(r'[./\w]+\.[\w]+', objective)[0]
    # Read the file content
    with open(file_path, 'r') as file:
        file_content = file.read()
    # Update the objective string to remove the file path
    objective = objective.split(file_path)[0].strip()
else:
    file_content = None

# Ask the user if they want to use search
use_search = input("Do you want to use search? (y/n): ").lower() == 'y'

task_exchanges = []
haiku_tasks = []

# Main execution loop
import asyncio

async def run_main_loop(objective, file_content, use_search):
    task_exchanges = []
    haiku_tasks = []

    while True:
        previous_results = [result for _, result in task_exchanges]
        if not task_exchanges:
            sub_tasks, search_queries, file_content_for_haiku = opus_orchestrator(objective, file_content, previous_results, use_search)
        else:
            sub_tasks, search_queries, _ = opus_orchestrator(objective, previous_results=previous_results, use_search=use_search)

        # Check if sub_tasks is empty to avoid IndexError
        if not sub_tasks:
            console.print("[bold red]No sub-tasks were returned by the orchestrator. Exiting the loop.[/bold red]")
            break  # or handle the situation appropriately

        if "The task is complete:" in sub_tasks[0]:
            final_output = sub_tasks[0].replace("The task is complete:", "").strip()
            break
        else:
            with ThreadPoolExecutor() as executor:
                sub_agent_futures = []
                loop = asyncio.get_event_loop()
                for i, sub_task in enumerate(sub_tasks):
                    search_query = search_queries[i] if search_queries and i < len(search_queries) else None
                    model = SUB_AGENT_MODELS[i % len(SUB_AGENT_MODELS)]
                    sub_agent_future = loop.run_in_executor(executor, enhanced_haiku_sub_agent, sub_task, search_query, model, use_search)
                    sub_agent_futures.append(sub_agent_future)

                sub_task_results = []
                for future in asyncio.as_completed(sub_agent_futures):
                    sub_task_result = await future
                    sub_task_results.append(sub_task_result)

                task_exchanges.append((sub_tasks, sub_task_results))

    # Create the .md filename
    sanitized_objective = re.sub(r'\W+', '_', objective)
    timestamp = datetime.now().strftime("%H-%M-%S")

    # Call Opus to review and refine the sub-task results
    refined_output = opus_refine(objective, [result for _, results in task_exchanges for result in results], timestamp, sanitized_objective)

    # Extract the project name from a match and sanitize it
    project_name_match = re.search(r'<project_name>(.*?)</project_name>', refined_output)
    project_name = project_name_match.group(1).strip() if project_name_match else sanitized_objective

    # Extract the folder structure from the refined output
    folder_structure_match = re.search(r'<folder_structure>(.*?)</folder_structure>', refined_output, re.DOTALL)
    folder_structure = {}
    if folder_structure_match:
        json_string = folder_structure_match.group(1).strip()
        try:
            folder_structure = json.loads(json_string)
        except json.JSONDecodeError as e:
            console.print(Panel(f"Error parsing JSON: {e}", title="[bold red]JSON Parsing Error[/bold red]", title_align="left", border_style="red"))
            console.print(Panel(f"Invalid JSON string: [bold]{json_string}[/bold]", title="[bold red]Invalid JSON String[/bold red]", title_align="left", border_style="red"))

    # Extract code files from the refined output
    code_blocks = re.findall(r'Filename: ([\w\.-]+).*?```.*?\n(.*?)```', refined_output, re.DOTALL)

    # Create the folder structure and code files
    create_folder_structure(project_name, folder_structure, code_blocks)

    # Truncate the sanitized_objective to a maximum of 50 characters
    max_length = 25
    truncated_objective = sanitized_objective[:max_length] if len(sanitized_objective) > max_length else sanitized_objective

    # Update the filename to include the project name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{truncated_objective}.md"

    # Prepare the full exchange log
    exchange_log = f"Objective: {objective}\n\n"
    exchange_log += "=" * 40 + " Task Breakdown " + "=" * 40 + "\n\n"
    for i, (prompts, results) in enumerate(task_exchanges, start=1):
        exchange_log += f"Round {i}:\n"
        for j, (prompt, result) in enumerate(zip(prompts, results), start=1):
            exchange_log += f"Task {i}.{j}:\n"
            exchange_log += f"Prompt: {prompt}\n"
            exchange_log += f"Result: {result}\n\n"
    exchange_log += "=" * 40 + " Refined Final Output " + "=" * 40 + "\n\n"
    exchange_log += refined_output

    # Write the exchange log to a file
    with open(filename, 'w') as file:
        file.write(exchange_log)

    # Display the saved exchange log file path
    console.print(f"\nFull exchange log saved to {filename}")

if __name__ == "__main__":


    # Extract the file path from the objective if it exists
    file_path = None
    if "./" in objective or "/" in objective:
        file_path = re.findall(r'[./\w]+\.[\w]+', objective)[0]
        objective = objective.replace(file_path, "").strip()

    # Read the file content if a file path is provided
    file_content = None
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()

    # Run the main loop with the collected inputs
    asyncio.run(run_main_loop(objective, file_content, use_search))

