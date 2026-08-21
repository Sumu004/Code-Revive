import json
import os
import shutil
import tempfile
import time
import zipfile
from tempfile import NamedTemporaryFile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ['GROQ_API_KEY'])

MAX_CONVERSION_ATTEMPTS = 5

NON_CODE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx', '.ppt', '.pptx')


def is_code_file(file_name):
    return not file_name.lower().endswith(NON_CODE_EXTENSIONS)


def extract_zip(zip_file, extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def create_directory_tree(root_dir):
    tree = {}
    for root, dirs, files in os.walk(root_dir):
        current_dir = tree
        for dir_name in root.split(os.sep)[1:]:
            current_dir = current_dir.setdefault(dir_name, {})
        for file_name in files:
            if not is_code_file(file_name):
                continue
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                with open(file_path, 'rb') as file:
                    content = file.read().decode('latin-1')
            current_dir[file_name] = content
    return tree


def analyze_code(code_contents):
    contents = ''
    for filename, code in code_contents.items():
        contents += f"\nCode from file '{filename}':"
        contents += f'\n\n{code}\n\n'
        contents += '-' * 20
    return contents


def extract_code_contents(json_data):
    code_contents = {}
    for directory, files in json_data.items():
        for filename, content in files.items():
            code_contents[filename] = content
    return code_contents


def extracting(zip_file_path):
    extract_dir = tempfile.mkdtemp()
    try:
        extract_zip(zip_file_path, extract_dir)
        json_data = create_directory_tree(extract_dir)
        code_contents = extract_code_contents(json_data)
        contents_of_code = analyze_code(code_contents)
        return contents_of_code, json_data
    except Exception as e:
        st.error(f'An error occurred while extracting the zip file: {e}')
        return None, None
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def getResponseAsJSON(prompt):
    return client.chat.completions.create(
        messages=[{'role': 'system', 'content': f'json {prompt}'}],
        model='llama3-8b-8192',
        response_format={'type': 'json_object'},
    )


def getResponse(prompt):
    return client.chat.completions.create(
        messages=[{'role': 'system', 'content': prompt}],
        model='llama3-8b-8192',
    )


def getContent(response):
    return response.choices[0].message.content


def checkWarning(response):
    text = response.choices[0].message.content.strip().lower()
    words = text.split()
    if not words:
        return None
    if words[0] == 'yes':
        return 1
    if words[0] == 'no':
        return 0
    return None


def analyze_codebase_with_llama(json_data, code_contents, desired_code):
    prompt_warning = construct_warning(code_contents, desired_code)
    response_warning = getResponse(prompt_warning)

    if checkWarning(response_warning):
        st.warning('Please select another code language.')
        st.stop()

    prompt_cc = construct_prompt_cc(code_contents, desired_code)
    prompt_test_case = construct_prompt_test_case(code_contents)
    prompt_documentation = construct_prompt_documentation(code_contents)

    response_doc = getResponse(prompt_documentation)
    response_cc = getResponse(prompt_cc)
    response_test_case = getResponse(prompt_test_case)

    response_convert = None
    acc_str = ''

    for attempt in range(MAX_CONVERSION_ATTEMPTS):
        prompt_convert = construct_prompt_convert(code_contents, getContent(response_cc))
        response_convert = getResponse(prompt_convert)

        prompt_for_test = construct_prompt_for_test(response_test_case, getContent(response_convert))
        response_test_result = getResponseAsJSON(prompt_for_test)

        data = json.loads(response_test_result.choices[0].message.content)
        acc = data['accuracy']
        acc_str = f'Accuracy: {acc}/10'

        if int(acc) >= 9:
            break

        time.sleep(4)
    else:
        st.warning(f'Could not reach a 9/10 accuracy after {MAX_CONVERSION_ATTEMPTS} attempts. Showing the best result found.')

    analyze_groq_response(response_doc, 'Technical Documentation')
    st.subheader(acc_str)
    analyze_groq_response(response_convert, 'Codes Converted')


def construct_warning(code_contents, desired_code):
    prompt = f"Can the Code contents be converted to {desired_code}.\n"
    prompt += f'Code Contents: {code_contents}. \n '
    prompt += f"The First word of the response should be 'Yes' if the Code Contents can't be converted to {desired_code}, 'No' if it can be converted.\n"
    prompt += 'Send a warning if it cannot be converted as a response.'
    return prompt


def construct_prompt_documentation(code_contents):
    prompt = 'Generate technical documentation based on the provided code snippet.\n'
    prompt += f'Code Contents: {code_contents}.\n'
    prompt += 'Return Technical Documentation on all codes.\n'
    prompt += 'DO IT FOR ALL CODES.\n'
    prompt += """ What is Technical Documentation?
Technical documentation encompasses any written material that elucidates the application, purpose, creation, or architecture of a product or service. Its primary objective is to elucidate aspects of what an organization offers. This documentation can take various forms, including how-to guides, user manuals, presentations, memos, reports, and more.

Who Creates Technical Documentation?
Technical documentation is typically crafted by technical writers, project managers, development team members, or subject matter experts. Various industries rely on technical documentation, including software, automotive, healthcare, and consumer products sectors.

Audience for Technical Documents
The audience for technical documents varies based on the document type. End users often interact with product-related documentation, while internal stakeholders and clients may engage with documentation pertaining to development processes, project progress, or technical specifications.

Importance of Technical Documentation
Technical documentation plays a pivotal role in facilitating understanding and usage of a product or service. For end users, it enables efficient product utilization and troubleshooting, potentially reducing the need for customer support. Internally, it enhances productivity and aligns teams by providing clear guidance and reference materials.

Types of Technical Documentation
Two primary categories of technical documentation are process documentation and user documentation. Process documentation delineates the development lifecycle of a product, while user documentation focuses on providing guidance to end users on product usage, troubleshooting, and features.

How to Create Technical Documentation
Have a Plan: Develop an outline detailing the components to include in the documentation.
Understand Your Audience: Tailor the documentation style and content to suit the intended audience's knowledge level and preferences.
Create a First Draft: Use templates or outlines to structure the document and include all necessary information.
Consider Adding Images: Visual aids such as diagrams and charts enhance understanding, especially for complex topics.
Review the Document: Solicit feedback from team members and subject matter experts to refine and update the document as needed.
Tips for Writing Technical Documentation
Be Consistent: Maintain uniformity in appearance, style, and tone throughout the document to enhance readability.
Be Concise: Edit the document to eliminate unnecessary or ambiguous content, focusing on conveying key information succinctly.
Optimize for Multiple Platforms: Ensure accessibility across various devices and platforms to maximize usability for the audience.
By adhering to these guidelines, you can create technical documentation that effectively communicates critical information and meets the needs of your audience.
                """
    return prompt


def construct_prompt_test_case(code_contents):
    prompt = f'Write test cases for all codes in the: \n {code_contents}\n'
    prompt += 'Atleast 10 cases for each code in all aspects'
    return prompt


def construct_prompt_cc(code_contents, desired_code):
    prompt = 'Analyse the provided code contents and determine the appropriate programming language for each file.\n'
    prompt += 'Understand the complete codebase and identify any dependencies or relationships between the code snippets.\n'
    prompt += '- The codebase is extracted from a directory structure stored in a JSON file.\n'
    prompt += '- It contains multiple code snippets representing various functionalities.\n\n'
    prompt += 'Your task is to:\n'
    prompt += '1. Identify the main funtionality of all programming languages used in the codebase.\n'
    prompt += f'2. Determine which files need to be converted to {desired_code}.\n'
    prompt += f'3. If conversion to {desired_code} is not feasible, suggest modern age coding languages as alternatives.\n\n'
    prompt += 'But all the codes must and should be instucted to be converted, and all the fountionality of the codes must be analysed and instructed correctly.\n'
    prompt += 'In your response, include:\n'
    prompt += '- A list of files and the recommended programming language for each.\n'
    prompt += '- Detailed instructions on how to convert each file to the recommended language.\n'
    prompt += '- Algorithms, methodologies, or tools required for the conversion.\n\n'
    prompt += 'Code Contents:\n'
    prompt += code_contents
    return prompt


def construct_prompt_convert(code_contents, convert_instructions):
    prompt = 'Analyse the following conversion instructions carefully:\n\n'
    prompt += f'{convert_instructions}\n\n'
    prompt += 'All the codes must and should be converted'
    prompt += 'Using the provided instructions, convert all code snippets accordingly.\n'
    prompt += '- Code Contents:\n'
    prompt += f'{code_contents}\n'
    prompt += '\n- Enclose the converted code with triple backticks (```) at the start and end.\n'
    prompt += '- Example:\n'
    prompt += '  ```code```\n'
    prompt += 'DO NOT ASK ANY QUESTIONS AND CONVERT THE CODES. Do not skip codes, all the codes must and should be converted, Only give the converted code not the previous code.\n'
    prompt += 'DO NOT ADD - Your code goes here, and so on, implement the code here, BUT YOU SHOULD COMPLETELY CONVERT ALL CODES WITHOUT LEAVEING ANYTHIB OUT.\n '
    return prompt


def construct_prompt_for_test(response_test_case, response_convert):
    prompt = 'It is in JSON FORMAT. Test all the test cases for each code and calculate the accuracy of the tests passed marked out of 10 based on the Converted code by automatically by yourself.\n'
    prompt += f'Converted code: {response_convert}\n'
    prompt += f'Test case: {response_test_case}\n'
    prompt += "Only one word response of the marks out of 10. and its keyword be 'accuracy'"
    prompt += r"{'accuracy':marks} <- in this format"
    return prompt


def analyze_groq_response(response, subhead):
    st.subheader(f'{subhead}:')
    st.write(response.choices[0].message.content)


def save_uploaded_file(uploaded_file):
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def main():
    st.title('CodeRevive')

    st.subheader('Upload Zip File')
    zip_file = st.file_uploader('Upload your zip file', type='zip')

    if not zip_file:
        st.warning('Please upload a zip file.')
        return

    zip_file_path = save_uploaded_file(zip_file)
    st.success('Zip file uploaded successfully!')

    st.sidebar.header('Settings')
    languages = ['Python', 'Java', 'C++', 'JavaScript', 'Go']
    custom_language = st.sidebar.text_input('Custom Language')
    selected_languages = st.sidebar.multiselect('Desired Code Languages', languages)
    desired_code = selected_languages if selected_languages else [custom_language] if custom_language else []

    if st.sidebar.button('Analyze Codebase'):
        contents_of_code, json_data = extracting(zip_file_path)
        if contents_of_code is None:
            return
        analyze_codebase_with_llama(json_data, contents_of_code, desired_code)


if __name__ == '__main__':
    main()
