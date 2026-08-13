# This section of the code handles page by page extraction from PDFs by converting PDFs into images,
# sending each image to the azure api and then returns the extracted content as Langchain document
# information
import base64
import concurrent
from concurrent.futures import ThreadPoolExecutor
import glob
import os
import re
from threading import current_thread
import time

from openai import AzureOpenAI, InternalServerError, APIStatusError
from langchain.schema import Document

import utils
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

# Loading the azure configuration
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_GPT_4o = os.environ.get("MODEL_GPT_4o")
MODEL_GPT_4o_MINI = os.environ.get("MODEL_GPT_4o_MINI")
API_VERSION = os.environ.get("API_VERSION")

# Creating a repeated AzureOpenAI client, in order to call for the chat completions endpoint.
# To minimise shared-state problems during parallel processing, a separate client is built for
# each worker thread. 
def init_openai_client(API_KEY: str):
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
    )

# A single page image's structured content is extracted, and Markdown formatting is used to try 
# to maintain tables. To prevent crashes while running in parallel, each worker thread chooses 
# its own OpenAI client. The function includes retry logic where it starts with a smaller model
# for cost, it falls back to the larger model on the final try if needed, and detects a known
# failure mode where the output becomes repetitive.
def extract_text_from_page_markdown(
    openai_clients, page_path: str, role_msg: str, instruction_msg: str
):
    thread = current_thread()
    thread_number = int(thread.name.split("_")[-1])
    opneai_client = openai_clients[thread_number]

    page_num = page_path.split("_p")[-1].split(".jpg")[0]
    file_basename = os.path.basename(page_path)

    b64_img = utils.encode_image(page_path)
    model_name = MODEL_GPT_4o_MINI
    attempts = 5
    while attempts > 0:
        try:
            response = opneai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": role_msg},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction_msg},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_img}"
                                },
                            },
                        ],
                    },
                ],
                temperature=0.0,
            )
            output = response.choices[0].message.content

            if (
                attempts > 1
                and detect_pattern_of_dots(output)
                or output
                == "I'm unable to extract text from the image directly. However, if you can provide the text or details from the image, I can help you analyze or summarize it."
            ):
                print(
                    f"The data was not extracted correctly from the page, attempts left: {attempts}"
                )
                raise ValueError(
                    f"The data was not extracted correctly from the page, attempts left: {attempts}"
                )

            output = remove_output_sys_messages(output)

            doc = Document(
                page_content=output,
                metadata={
                    "source": file_basename,
                    "page_num": page_num,
                    "usage": response.usage,
                },
            )
            os.remove(page_path)
            return doc

        except:
            attempts -= 1
            if attempts == 1:
                model_name = MODEL_GPT_4o
    else:
        return None

# Same Markdown extractor but needed when extraction behaviour is switched later. 
def extract_text_from_page(
    openai_clients, page_path: str, role_msg: str, instruction_msg: str
):
    thread = current_thread()
    thread_number = int(thread.name.split("_")[-1])
    opneai_client = openai_clients[thread_number]

    page_num = page_path.split("_p")[-1].split(".jpg")[0]
    file_basename = os.path.basename(page_path)

    b64_img = utils.encode_image(page_path)
    model_name = MODEL_GPT_4o_MINI
    attempts = 5
    while attempts > 0:
        try:
            response = opneai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": role_msg},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction_msg},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_img}"
                                },
                            },
                        ],
                    },
                ],
                temperature=0.0,
            )

            output = response.choices[0].message.content

            if (
                attempts > 1
                and detect_pattern_of_dots(output)
                or output
                == "I'm unable to extract text from the image directly. However, if you can provide the text or details from the image, I can help you analyze or summarize it."
            ):
                print(
                    f"The data was not extracted correctly from the page, attempts left: {attempts}"
                )
                raise ValueError(
                    f"The data was not extracted correctly from the page, attempts left: {attempts}"
                )

            output = remove_output_sys_messages(output)

            doc = Document(
                page_content=output,
                metadata={
                    "source": file_basename,
                    "page_num": page_num,
                    "usage": response.usage,
                },
            )
            os.remove(page_path)
            return doc

        except:
            attempts -= 1
            if attempts == 1:
                model_name = MODEL_GPT_4o
    else:
        return None

# Splitting a list into chunks.
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

# Defining the parallel text extraction which runs the page across multiple threads, so long
# documents can be processed quickly. This creates one Azure client per worker, submits
# extraction jobs to a thread pool, and collects the results and returns them as a 
# document.
def parallel_text_extraction_markdown(API_KEY: str, img_paths: list[str], workers: int):

    role_msg = """
            You are a professional image-to-text document parser.
            Extract all text from these images, treating them as pages of a PDF document.
            Try to format any tables found in the images.
            Do not include page numbers, page headers, or page footers.
            Return only the extracted text.  No commentary.
            **Exclude Nothing**: The only text that does not need to be included is the page number.
            Return the text as a Markdown document.
                    1. Keep the source language.
                    2. Include headers and footers.
                    3. Don't interpolate or make up data.
                    4. Extract all text and tables as Markdown
                    5. Here's a comprehensive list of Markdown formatting guidelines that cover various aspects of text formatting, lists, tables, links, images, and more:

                            1. **Headings**: Use `#` for headings. The number of `#` symbols indicates the heading level (1-6).
                               - Example:
                                 - `# Heading 1`
                                 - `## Heading 2`
                                 - `### Heading 3`

                            2. **Bold Text**: Use double asterisks `**` or double underscores `__` to make text bold.
                               - Example: `**bold text**` or `__bold text__`

                            3. **Italic Text**: Use single asterisks `*` or single underscores `_` to italicize text.
                               - Example: `*italic text*` or `_italic text_`

                            4. **Strikethrough**: Use double tildes `~~` to strikethrough text.
                               - Example: `~~strikethrough~~`

                            5. **Blockquotes**: Use the `>` character to create blockquotes.
                               - Example:
                                 ```
                                 > This is a blockquote.
                                 ```

                            6. **Unordered Lists**: Use asterisks `*`, plus `+`, or hyphens `-` to create unordered lists.
                               - Example:
                                 ```
                                 - Item 1
                                 - Item 2
                                   - Subitem 2.1
                                 ```

                            7. **Ordered Lists**: Use numbers followed by a period to create ordered lists.
                               - Example:
                                 ```
                                 1. First item
                                 2. Second item
                                    1. Subitem 2.1
                                 ```

                            8. **Links**: Use square brackets for the link text and parentheses for the URL.
                               - Example: `[Link text](http://example.com)`

                            9. **Images**: Use an exclamation mark `!` followed by square brackets for the alt text and parentheses for the image URL.
                                - Example: `![Alt text](http://example.com/image.jpg)`

                            10. **Horizontal Rules**: Use three or more hyphens `---`, asterisks `***`, or underscores `___` to create horizontal rules.
                                - Example:
                                  ```
                                  ---
                                  ```

                            11. **Line Breaks**: End a line with two or more spaces and press Enter, or use `<br>` for a line break.

                            12. **Footnotes**: Use square brackets with a caret `[^1]` for footnotes, and define them at the bottom.
                                - Example:
                                  ```
                                  This is a sentence with a footnote.[^1]

                                  [^1]: This is the footnote text.
                                  ```

                            13. **Escaping Characters**: Use a backslash to escape special Markdown characters.
                                - Example: `not italic`

                            14. **Task Lists**: Use `- [ ]` for incomplete tasks and `- [x]` for completed tasks.
                                - Example:
                                  ```
                                  - [ ] Task 1
                                  - [x] Task 2
                                  ```

                            15. **Nested Lists**: Indent sub-items with two spaces or a tab.
                                - Example:
                                  ```
                                  - Item 1
                                    - Subitem 1.1
                                  ```
                    6. For tables, Markdown require a specific format to be rendered correctly:
                                1. Header Separator Consistency: Ensure that the header separator row contains the same number of columns as the header row. Use hyphens (-) to create the separator.
                                2. Column Alignment: Maintain proper alignment of columns in both the header and data rows to enhance readability, even if explicit alignment is not required by Markdown.
                                3. Uniform Column Count: Verify that each data row has the same number of cells as the header row to prevent misalignment and confusion.
                                4. Cell Spacing: Include a space before and after the content within each cell to improve readability.
                                5. Consistent Data Formatting: Use a uniform format for similar types of data (e.g., currency, percentages) throughout the table to ensure clarity.
                                6. Header Repetition: If the table extends beyond a single page, ensure that the table header is repeated on each page for easier reference.
                                7. Avoid Extra Columns: Check for and eliminate any extra columns in the header or data rows that do not align with the intended structure of the table.

                    7. Make sure to include all amounts in euros or dollars, percentages (%), addresses, dates, numbers, and tables.

                    """

    instruction_msg = (
        "Extract all text, tables, footnotes, and citations from the image"
    )

    MAX_NUM_WORKERS = 15

    if workers > MAX_NUM_WORKERS:
        raise ValueError(
            f"Max number of workers exceeded. The max number of workers is {MAX_NUM_WORKERS}; this prevents reaching the rate limits"
        )

    workers = min(workers, len(img_paths))

    openai_clients_list = [init_openai_client(API_KEY) for _ in range(workers)]
    processes = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for page in img_paths:
            input_data = {
                "openai_clients": openai_clients_list,
                "page_path": page,
                "role_msg": role_msg,
                "instruction_msg": instruction_msg,
            }
            processes.append(
                executor.submit(extract_text_from_page_markdown, **input_data)
            )

        results = concurrent.futures.as_completed(processes)

    results = [r.result() for r in results]
    return results

# Converts a PDF into page images, transfers them through the parallel extraction system,
# and returns a list of page-level documents.
def extract_text_from_file_markdown(
    file_path: str, API_KEY: str, workers: int = 1, temp_folder: str = None
) -> list[Document]:

    file_basename = utils.pdf_to_image(
        file_path, temp_folder=temp_folder
    )
    img_paths = glob.glob(f"{temp_folder}/tmp_*")
    results = parallel_text_extraction_markdown(API_KEY, img_paths, workers)

    return results

# This function flags the failed extractions that return long repetitive sequences so that
# the page can be retried.
def detect_pattern_of_dots(text):
    pattern = r"(. . . .)"
    pattern = rf"({'. . . . ' * 30})"
    matches = re.findall(pattern, text)
    if len(matches) > 1:
        print(f"Dots detected, the pattern repeats {len(matches)} times")
        return True
    else:
        return False

# The system here is removing unnecessary text that is retrieved with the data so that
# only the data is extracted.
def remove_output_sys_messages(text):
    sys_messages = [
        "### Transcription of the Document",
        "If you need further assistance or details, feel free to ask!",
        "Here is the transcription of the text from the document:",
        "Here is the text extracted from the image:",
        "Let me know if you need any further assistance!",
        "There are no diagrams, tables, plots, or charts present in the image.",
        "This text includes all relevant details, including names, numbers, and descriptions from the document. Here is the extracted text from the image:",
        "There are no diagrams, tables, plots, or charts in the image. Here is the extracted text from the image:",
        "This text includes all the relevant details from the document. Here is the extracted text from the image:",
        "This text includes all the relevant information from the document, including the details of the insured properties, coverage limits, and deductibles. Here is the extracted text from the image:",
        "This text includes all the relevant information from the document. Here is the extracted text from the image:",
        "No diagrams, tables, plots, or charts are present in the image.",
        "This text includes all relevant details from the document. If you need further assistance, feel free to ask! Here is the extracted text from the image:",
        "Here is the extracted text from the image:",
        "This text includes all the relevant details from the document. If you need further assistance, feel free to ask!",
        "This text includes all the relevant information from the document.",
        "",
        "If you need further assistance, feel free to ask!",
        "",
        "This text includes all relevant details, including names, amounts, and descriptions from the document.",
        "",
        "**Referencia a 'ajuar':** No se encontró ninguna referencia a 'ajuar' en el texto extraído.",
        "No references to 'ajuar' were found in the text.",
        'This text includes all relevant information, including references to "ajuar" and any amounts or percentages found in the original document.',
        'No specific references to "ajuar", "prima", "limites", or "sumas aseguradas" were found in the text.',
        "```markdown",
        "```",
        "I'm unable to extract text from the image directly. However, if you can provide the text or details from the document, I can help you format it or analyze it as needed.",
        "No references to 'ajuar', 'prima', 'limites', 'sumas aseguradas', or 'tomador del seguro' were found.",
        "This text includes all relevant information as requested.",
        "Extracted Text",
        "Here is the extracted text from the image, formatted in Markdown:",
        "markdown",
        "I'm unable to extract text from images directly. If you can provide the text or details from the image, I'd be happy to help format it or assist you with any specific requests!",
    ]
    for message in sys_messages:
        text = text.replace(message, "")
    return text