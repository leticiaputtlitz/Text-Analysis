from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import io
import base64
from datasets import load_dataset
from deep_translator import GoogleTranslator
import dash
from metrics.sentiment import SentimentAnalyzer
from metrics.toxicity import ToxicityAnalyzer
from metrics.prompt_analyses import PromptAnalyzer
from metrics.refusal import RefusalAnalyzer
from metrics.topics import TopicsAnalyzer
from metrics.patterns import RegexAnalyzer
from metrics.pii import PIIAnalyzer
from metrics.textstat import TextStatAnalyzer
from presidio_analyzer import AnalyzerEngine
import plotly.express as px
import time
import dash_bootstrap_components as dbc
from collections import Counter
import dash_tour_component

analyzer = AnalyzerEngine()


# Initialize the app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    html.Footer(
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.P([
                        "Developed by Letícia Puttlitz - ",
                        html.A("GitHub", href="https://github.com/leticiaputtlitz", target="_blank")
                    ], style={'textAlign': 'center', 'padding': '10px', 'color': '#6c757d'})
                )
            ),
            fluid=True,
            style={'borderTop': '1px solid #e0e0e0', 'paddingTop': '10px', 'marginTop': '20px'}
        )
    ),
    dash_tour_component.DashTour(
        steps=[
            {
                'selector': '[id="upload-data"]',
                'content': "Upload your dataset here. The dataset should contain three columns: one named 'instruction' for the instructions, another named 'input' or 'prompt' for the inputs, and a third named 'response' or 'output' for the responses. The 'instruction' column is optional."
            },
            {
                'selector': '[id="df-hf"]',
                'content': 'Enter the Hugging Face dataset path here.',
            },
            {
                'selector': '[id="manual-prompt"]',
                'content': 'Use this field to manually enter a custom prompt, instead of uploading a dataset.',
            },
            {
                'selector': '[id="manual-response"]',
                'content': 'Use this field to manually enter a custom response for the prompt you provided, instead of uploading a dataset.',
            },
            {
                'selector': '[id="instruction-name"]',
                'content': 'If you upload a dataset and the column for instructions is not named "instruction", enter its current name here, and it will be automatically renamed.',
            },
            {
                'selector': '[id="input-name"]',
                'content': 'If you upload a dataset and the column for instructions is not named "input" or "prompt", enter its current name here, and it will be automatically renamed.',
            },
            {
                'selector': '[id="response-name"]',
                'content': 'If you upload a dataset and the column for instructions is not named "output" or "response, enter its current name here, and it will be automatically renamed.',
            },
            {
                'selector': '[id="process-button"]',
                'content': 'Click here to process the dataset.',
            },
            {
                'selector': '[id="link-data-view"]',
                'content': 'View the uploaded dataset here.',
            },
            {
                'selector': '[id="link-sentiment"]',
                'content': 'Perform sentiment analysis here.',
            },
            {
                'selector': '[id="link-toxicity"]',
                'content': 'Perform toxicity analysis here.',
            },
            {
                'selector': '[id="link-prompt"]',
                'content': 'Analyze prompts for jailbreaks and injections here.',
            },
            {
                'selector': '[id="link-refusal"]',
                'content': 'Analyze refusals here.',
            },
            {
                'selector': '[id="link-topics"]',
                'content': 'Perform topic analysis here.',
            },
            {
                'selector': '[id="link-patterns"]',
                'content': 'Perform pattern analysis here.',
            },
            {
                'selector': '[id="link-entity-recognition"]',
                'content': 'Perform private entity recognition here.',
            },
            {
                'selector': '[id="link-textstat"]',
                'content': 'Perform text statistics analysis here.',
            }
            

        ],
        isOpen=False,
        id="tour_component"
    ),
    
    # Botão para iniciar o tour
    html.Div([
        html.Button(
            "Welcome! Click here to start the tour",
            id='open_tour_button',
            style={
                'fontSize': '20px',
                'padding': '15px 30px',
                'backgroundColor': '#transparent',  # Cor cinza
                'color': '#6c757d',
                'border': 'none',
                'borderRadius': '8px',
                'cursor': 'pointer',
                'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.2)',
                'marginRight': '10px'
            }
        ),
        html.Button(
            "x",
            id='close_tour_button',
            style={
                'fontSize': '16px',
                'width': '24px',      # Largura igual à altura para formar um círculo
                'height': '24px', 
                'backgroundColor': '#transparent',  # Cor vermelha para o botão de fechar
                'color': '#6c757d',
                'border': 'none',
                'borderRadius': '50%',
                'cursor': 'pointer',
                'position': 'absolute',
                'top': '-5px',  # Posiciona o botão no canto superior direito
                'right': '-5px',
                'display': 'flex',            # Usa flexbox
                'alignItems': 'center',        # Centraliza verticalmente
                'justifyContent': 'center',    # Centraliza horizontalmente
                'padding': '0'
            }
        )
    ], style={'position': 'fixed', 'top': '20px', 'right': '20px', 'zIndex': '1000', 'display': 'flex', 'alignItems': 'center'}),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Text Analysis Studio"),
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Dataset Upload", style={'textAlign': 'center'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Upload a file", style={'marginBottom': '10px', 'fontSize': '15px'}),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Button('Upload'),
                                    multiple=False,
                                    accept='.csv, .xlsx, .json, .tsv'
                                ),
                                html.Div(id='upload-status', style={'marginTop': '10px', 'fontSize': '10px'})
                            ], width=4),
                            dbc.Col(
                                html.Div([
                                    html.Div("OR", style={'textAlign': 'center', 'fontSize': '15px', 'marginBottom': '10px'}),
                                    html.Div(style={'borderLeft': '2px solid #000', 'height': '80px', 'width': '0px', 'margin': '0 auto'})
                                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
                                width=1
                            ),
                            dbc.Col([
                                html.Label("Upload from Hugging Face", style={'marginBottom': '10px', 'fontSize': '15px'}),
                                dcc.Input(
                                    id='df-hf',
                                    type='text',
                                    placeholder='Enter HF dataset path',
                                    style={'width': '100%'}
                                ),
                                html.Div(id='hf-upload-status', style={'marginTop': '10px', 'fontSize': '10px'})
                            ], width=7),
                        ], align='center'),
                    ])
                ], style={'padding': '0px', 'marginBottom': '20px'}),
            ], width=4),
            # Adicione um novo Card para inserção manual de Prompt e Response
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Manual Input", style={'textAlign': 'center'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Label("Prompt:", style={'marginRight': '10px', 'fontSize': '17px', 'textAlign': 'right'}),
                                    dcc.Input(id='manual-prompt', type='text', placeholder='Enter prompt', style={'width': '70%'})
                                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end', 'marginBottom': '10px'}),
                            ], width=12),
                            dbc.Col([
                                html.Div([
                                    html.Label("Response:", style={'marginRight': '10px', 'fontSize': '17px', 'textAlign': 'right'}),
                                    dcc.Input(id='manual-response', type='text', placeholder='Enter response', style={'width': '70%'})
                                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end', 'marginBottom': '10px'}),
                            ], width=12),
                        ])
                    ])
                ], style={'padding': '0px', 'marginBottom': '20px'}),
            ], width=4),


            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Column Names", style={'textAlign': 'center'}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Label(id='instruction-label', children="Instruction:", style={'fontSize': '17px', 'marginRight': '10px'}),
                                    dcc.Input(id='instruction-name', type='text', placeholder='Enter column name', style={'width': '58%', 'marginLeft': '10px'}),
                                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end', 'marginBottom': '10px'}),
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Label(id='input-label', children="Input/Prompt:", style={'fontSize': '17px', 'marginRight': '10px'}),
                                    dcc.Input(id='input-name', type='text', placeholder='Enter column name', style={'width': '59%', 'marginLeft': '10px'}),
                                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end', 'marginBottom': '10px'}),
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Label(id='response-label', children="Output/Response:", style={'fontSize': '17px', 'marginRight': '10px'}),
                                    dcc.Input(id='response-name', type='text', placeholder='Enter column name', style={'width': '60%', 'marginLeft': '10px'}),
                                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end', 'marginBottom': '10px'}),
                            ])
                        ])
                    ])
                ], style={'padding': '0px', 'marginBottom': '20px'}),
            ], width=4),
        ], align='center'),
        dbc.Row([
            dbc.Col([
                html.Button('Process', id='process-button', n_clicks=0, style={'width': '100%'}),
            ], width=12),
        ], style={'marginTop': '20px', "marginBottom": "20px"}),
        dcc.Loading(id="loading", type="circle", children=[html.Div(id='output-data-upload', style={'marginTop': '5px'})]),
        dcc.Store(id='stored-dataframe'),
        dcc.Store(id='sentiment-results'),
        dcc.Store(id='toxicity-results'),
        dcc.Store(id='prompt-results'),
        dcc.Store(id='refusal-results'),
        dcc.Store(id='topics-results'),
        dcc.Store(id='patterns-results'),
        dcc.Store(id='entity-results'),
        dcc.Store(id='textstat-results'),
        dcc.Store(id='dataset-loaded', data=False),  # Store para verificar se o dataset foi carregado

        # Interval for progress tracking
        dcc.Interval(id="progress-interval", n_intervals=0, interval=2000, disabled=True),  # Intervalo ajustado para 1000ms (1 segundo)
        dcc.Interval(id="sentiment-progress-interval", n_intervals=0, interval=2000, disabled=True),
        dcc.Interval(id="prompt-progress-interval", n_intervals=0, interval=2500, disabled=True),
        dcc.Interval(id="refusal-progress-interval", n_intervals=0, interval=2000, disabled=True),
        dcc.Interval(id="topics-progress-interval", n_intervals=0, interval=4000, disabled=True), 
        dcc.Interval(id="patterns-progress-interval", n_intervals=0, interval=2500, disabled=True), 
        dcc.Interval(id="entity-progress-interval", n_intervals=0, interval=3000, disabled=True),
        dcc.Interval(id="textstat-progress-interval", n_intervals=0, interval=3000, disabled=True), 

        # Sidebar for navigation
        dbc.Row([
            dbc.Col([
                dbc.Nav([
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Dataset View", href="#", id='link-data-view', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}  # Diminui o padding interno
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}  # Torna o card menor e ajusta o espaçamento entre eles
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Sentiment", href="#", id='link-sentiment', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Toxicity", href="#", id='link-toxicity', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Jailbreak and Injection", href="#", id='link-prompt', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Refusal", href="#", id='link-refusal', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Topics", href="#", id='link-topics', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Pattern", href="#", id='link-patterns', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Private Entity Recognition", href="#", id='link-entity-recognition', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.NavLink("Text Statistics", href="#", id='link-textstat', className="nav-link", style={'color': 'black'}),
                            style={'padding': '5px'}
                        ),
                        style={'marginBottom': '5px', 'backgroundColor': '#e0e0e0', 'width': '150px'}
                    ),
                ], vertical=True, pills=True, style={'padding': '10px', 'border': '1px solid #ddd'}),
            ], width=2),


            dbc.Col([
                html.Div(id='main-content', children=[
                    html.H3('Tool Overview'),
                    html.P('This tool offers a comprehensive suite of tools for analyzing your dataset. Select an option from the sidebar to begin. Below is a brief description of each analysis available:'),
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Dataset View'))),
                                dbc.CardBody([
                                    html.P('Displays the uploaded dataset in a tabular format, allowing you to inspect the data before running analyses.')
                                ])
                            ]), width=12, style={'marginBottom': '20px'}
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Sentiment'))),
                                dbc.CardBody([
                                    html.P('The Sentiment Analysis tool evaluates the emotional tone of the text in your dataset by applying natural language processing techniques. This analysis is performed on both the "prompt" and "response" columns using the NLTK Vader sentiment analyzer. Each piece of text is assigned a compound sentiment score ranging from -1 to 1, where -1 indicates the most negative sentiment and 1 indicates the most positive sentiment. This score helps to gauge the overall sentiment expressed in the dataset, providing insights into how positive, negative, or neutral the text is.'),
                                    html.P([
                                        "For more information on sentiment analysis using NLTK, visit the ",
                                        html.A('NLTK Sentiment', href='https://www.nltk.org/howto/sentiment.html', target='_blank'),
                                        "."
                                    ])
                                ])
                            ]), width=12, style={'marginBottom': '20px'}
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Toxicity'))),
                                dbc.CardBody([
                                    html.P('The Toxicity Analysis tool evaluates the level of toxicity present in the prompts and responses of your dataset. This analysis is essential for identifying potentially harmful, offensive, or abusive content. The toxicity score is calculated using HuggingFace\'s "martin-ha/toxic-comment-model" toxicity analyzer, which is a state-of-the-art model trained to detect toxic language in various forms of text. Each piece of text is assigned a toxicity score ranging from 0 to 1, where 0 indicates no toxicity and 1 indicates maximum toxicity. By analyzing the toxicity of the text, this tool helps in moderating content, ensuring safe and respectful communication, and understanding the presence of harmful language patterns in user-generated content or conversational AI responses.'),
                                    html.P([
                                        "For more details, visit the ",
                                        html.A('martin-ha/toxic-comment-model', href='https://huggingface.co/martin-ha/toxic-comment-model', target='_blank'),
                                        " page."
                                    ])
                                ])
                            ]), width=12, style={'marginBottom': '20px'}
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Jailbreak and Injection'))),
                                dbc.CardBody([
                                    html.P("This tool detects jailbreaks and injections in prompts, which are attempts to bypass model restrictions or inject unintended commands into the model's behavior. It works by analyzing the similarity between user inputs and known examples of jailbreak attempts and harmful behaviors."),
                                    html.P('The prompt injection metric returns the maximum similarity score between a given prompt and a database of known jailbreak attempts and harmful behaviors. This database is managed using the FAISS package, which allows for efficient similarity search. A higher score indicates a closer match to a known jailbreak attempt or harmful behavior, suggesting that the prompt may be trying to manipulate the model.'),
                                    html.P("This analysis is similar to the jailbreak metric from the themes module, but with some differences. The similarity scores in both analysis are calculated by determining the cosine similarity between text embeddings, generated using HuggingFace's 'sentence-transformers/all-MiniLM-L6-v2' model"),
                                    html.P('By analyzing these similarities, the tool helps in identifying and mitigating potentially harmful or manipulative inputs in user-generated content or conversational AI prompts, enhancing the safety and reliability of the model.'),
                                    html.P([
                                        "For more details on the model used, visit the ",
                                        html.A('sentence-transformers/all-MiniLM-L6-v2', href='https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2', target='_blank'),
                                        " page."
                                    ]),
                                    html.P([
                                        "For more information on the LangKit library used for this analysis, please refer to the ",
                                        html.A('LangKit Documentation', href='https://github.com/whylabs/langkit/blob/main/langkit/docs/modules.md', target='_blank'),
                                        "."
                                    ])
                                ])
                            ]), width=12, style={'marginBottom': '20px'}
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Refusal'))),
                                dbc.CardBody([
                                    html.P("The Refusal Analysis tool examines the responses generated by the model to identify instances where it refuses to answer or carry out a request. This is crucial for understanding how the model handles questions or commands that might be outside its intended functionality or ethical boundaries."),
                                    html.P("The refusal score for this analysis is determined by calculating the cosine similarity between the embeddings generated from the target text and a collection of predefined refusal examples. These embeddings are created using HuggingFace's 'sentence-transformers/all-MiniLM-L6-v2' model, which excels at producing dense vector representations of text."),
                                    html.P("The analysis returns the highest similarity score found among all examples."),
                                    html.P([
                                        "To learn more about the model utilized in this analysis, please visit the ",
                                        html.A('sentence-transformers/all-MiniLM-L6-v2', href='https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2', target='_blank'),
                                        " page."
                                    ]),
                                    html.P([
                                        "To find out more about the LangKit library employed for this analysis, please see the ",
                                        html.A('LangKit Documentation', href='https://github.com/whylabs/langkit/blob/main/langkit/docs/modules.md', target='_blank'),
                                        "."
                                    ])
                                ])
                            ]), width=12, style={'marginBottom': '20px'}
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Topics'))),
                                dbc.CardBody([
                                    html.P('The Topics Analysis tool performs topic modeling on the dataset, effectively categorizing text into various topics based on content similarity. This allows for a better understanding of the main subjects discussed within the dataset. Users have the flexibility to either specify a list of topics they are interested in or let the system automatically determine the most relevant topics.'),
                                    html.P('The tool utilizes the "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" model from HuggingFace, a powerful model designed for multilingual text classification. This model is used to assign each piece of text to one of the predefined topics based on its content. The default topics include law, finance, medical, education, politics, and support.'),
                                    html.P('By analyzing the text with this model, the tool calculates scores for each topic and assigns the text to the topic with the highest score. This process helps in identifying the dominant themes and subjects within the dataset, allowing for a more structured and insightful analysis of the text.'),
                                    html.P([
                                        "For more details on the model used for topic classification, visit the ",
                                        html.A('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7', href='https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7', target='_blank'),
                                        " page."
                                    ])
                                ])
                            ]), 
                            width=12, 
                            style={'marginBottom': '20px'}
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Pattern'))),
                                dbc.CardBody([
                                    html.P('The Pattern Analysis tool is designed to identify specific patterns within the dataset using regular expressions (regex). This functionality is particularly useful for detecting structured or repeated content.'),
                                    html.P("The use of the Pattern Analysis tool is crucial for identifying sensitive personal information such as phone numbers, email addresses, credit card numbers, and identification documents. Detecting these patterns is essential to prevent data leakage, which occurs when confidential information is inadvertently included in datasets or reports. By identifying and flagging this information, the tool helps ensure that shared or analyzed data complies with privacy and security regulations, protecting individuals' privacy."),
                                    html.P('Currently supported patterns include:'),
                                    html.Ul([
                                        html.Li('US Social Security Numbers (SSN)'),
                                        html.Li('US and BR Credit Card Numbers'),
                                        html.Li('US and BR Phone Numbers'),
                                        html.Li('US and BR Mailing Addresses'),
                                        html.Li('Email Addresses'),
                                        html.Li('Brazilian CPF and CNPJ numbers'),
                                        html.Li('Brazilian Postal Codes (CEP)'),
                                    ]),
                                ])
                            ]), 
                            width=12, 
                            style={'marginBottom': '20px'}
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Private Entity Recognition'))),
                                dbc.CardBody([
                                    html.P('The Private Entity Recognition tool identifies and categorizes various types of sensitive private entities within your dataset, such as personal information, locations, and more. This functionality is critical for ensuring data privacy and compliance with regulations by preventing data leakage of sensitive information.'),
                                    html.P('The tool utilizes Microsoft\'s Presidio as its engine for identifying Personally Identifiable Information (PII). Presidio is a highly flexible and customizable platform designed for the detection of sensitive data. It currently supports identifying a comprehensive list of entities, including: Tax File Numbers (AU_TFN), Credit Card Numbers, International Bank Account Numbers (IBAN), U.S. Bank Numbers, U.S. Individual Taxpayer Identification Numbers (US_ITIN), Email Addresses, Personal Names, Phone Numbers, Australian Business Numbers (AU_ABN), Australian Company Numbers (AU_ACN), Indian Permanent Account Numbers (IN_PAN), Indian Vehicle Registrations, U.S. Driver Licenses, IP Addresses, Locations, Bank Account Numbers, Cryptocurrency Wallets, Date and Time Stamps, Medical Licenses, Organizations, UK National Health Service Numbers (UK_NHS), and URLs.'),
                                    html.P('By detecting these entities, the tool helps prevent data leakage, which is the unauthorized disclosure of sensitive information. Ensuring the identification and proper handling of PII is essential for maintaining the privacy and security of individuals and minimizing legal and financial risks for organizations.'),
                                    html.P([
                                        "For more information on Microsoft's Presidio used for this analysis, please refer to the ",
                                        html.A('Presidio Documentation', href='https://github.com/microsoft/presidio/', target='_blank'),
                                        "."
                                    ])
                                ])
                            ]), width=12, style={'marginBottom': '20px'}
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5(html.B('Text Statistics'))),
                                dbc.CardBody([
                                    html.P('The Text Statistics Analysis tool evaluates the readability and complexity of text within the dataset, providing various standard metrics to assess the ease or difficulty of reading the content. This analysis is particularly useful for understanding the accessibility of the text for different audiences and ensuring it aligns with the intended readability level.'),
                                    html.P('Key metrics provided by TextStat Analysis include:'),
                                    html.Ul([
                                        html.Li([
                                            html.B('Flesch Reading Ease:'),
                                            ' This score measures the readability of the text. The higher the score, the easier it is to understand the text. A score between 90-100 is considered very easy to read, while a score between 0-29 is considered very confusing.'
                                        ]),
                                        html.Li([
                                            html.B('SMOG Index:'),
                                            ' The SMOG Index estimates the years of education a person needs to understand a piece of writing. It is particularly useful for assessing the readability of more complex texts.'
                                        ]),
                                        html.Li([
                                            html.B('Flesch-Kincaid Grade Level:'),
                                            ' This metric indicates the U.S. school grade level required to understand the text. For example, a score of 9.3 suggests that a ninth grader would be able to read the document.'
                                        ]),
                                        html.Li([
                                            html.B('Coleman-Liau Index:'),
                                            ' Similar to other readability scores, the Coleman-Liau Index determines the grade level of the text but focuses on the number of letters per 100 words and the number of sentences per 100 words.'
                                        ]),
                                        html.Li([
                                            html.B('Automated Readability Index (ARI):'),
                                            ' ARI provides an estimate of the U.S. school grade level required to comprehend the text, based on characters per word and words per sentence.'
                                        ]),
                                        html.Li([
                                            html.B('Dale-Chall Readability Score:'),
                                            ' This score uses a list of 3,000 common words to assess readability, making it particularly effective for texts intended for younger audiences.'
                                        ]),
                                        html.Li([
                                            html.B('Difficult Words:'),
                                            ' This metric counts the number of words that are not common or easily understandable by readers below college level.'
                                        ]),
                                        html.Li([
                                            html.B('Linsear Write Formula:'),
                                            ' This formula is used to calculate the U.S. school grade level of the text based on the number of easy and difficult words.'
                                        ]),
                                        html.Li([
                                            html.B('Gunning Fog Index:'),
                                            ' The Gunning Fog Index estimates the years of formal education a reader needs to understand the text on the first reading. A score of 12 indicates high school level reading difficulty.'
                                        ]),
                                        html.Li([
                                            html.B('Text Standard:'),
                                            ' Based on multiple readability tests, this metric provides an overall estimation of the grade level required to understand the text.'
                                        ]),
                                        html.Li([
                                            html.B('Lexicon Count:'),
                                            ' This measures the number of words in the text, providing insight into the length and complexity of the content.'
                                        ]),
                                        html.Li([
                                            html.B('Sentence Count:'),
                                            ' This counts the number of sentences in the text, which can be indicative of the text’s complexity and structure.'
                                        ]),
                                        html.Li([
                                            html.B('Syllable Count:'),
                                            ' The number of syllables in the text can impact readability, with more syllables indicating more complex words.'
                                        ]),
                                        html.Li([
                                            html.B('Character Count:'),
                                            ' This measures the total number of characters in the text, offering another view of text length.'
                                        ]),
                                        html.Li([
                                            html.B('Polysyllable Count:'),
                                            ' This counts the number of words with three or more syllables, which typically indicates higher reading difficulty.'
                                        ]),
                                        html.Li([
                                            html.B('Monosyllable Count:'),
                                            ' This counts the number of one-syllable words, often associated with easier readability.'
                                        ])
                                    ]),
                                    html.P('These metrics help in understanding the readability and complexity of text, making it easier to tailor content to the appropriate audience and ensure compliance with readability standards.'),
                                    html.P([
                                        "For more detailed information on the TextStat library used for this analysis, please visit the ",
                                        html.A('TextStat Documentation', href='https://pypi.org/project/textstat/', target='_blank'),
                                        "."
                                    ])
                                ])
                            ]), width=12, style={'marginBottom': '20px'}
                        )

                    ])
                ])
            ], width=10)


        ], align='start')
    ]),
    html.Footer(
        dbc.Container(
            dbc.Row(
                dbc.Col(
                    html.P([
                        "Developed by Letícia Puttlitz - ",
                        html.A("GitHub", href="https://github.com/leticiaputtlitz", target="_blank")
                    ], style={'textAlign': 'center', 'padding': '10px', 'color': '#6c757d'})
                )
            ),
            fluid=True,
            style={'borderTop': '1px solid #e0e0e0', 'paddingTop': '10px', 'marginTop': '20px'}
        )
    ),
])

# Callback para abrir o tour
@app.callback(
    Output('tour_component', 'isOpen'),
    [Input('open_tour_button', 'n_clicks')],
    prevent_initial_call=True
)
def open_tour_component(value):
    return True

# Callback para esconder o botão de tour
@app.callback(
    Output('open_tour_button', 'style'),
    Output('close_tour_button', 'style'),
    [Input('close_tour_button', 'n_clicks')],
    prevent_initial_call=True
)
def close_tour_component(n_clicks):
    if n_clicks:
        # Retorna estilos que escondem os botões
        return {'display': 'none'}, {'display': 'none'}
    return dash.no_update, dash.no_update

# Function to process the uploaded file
def parse_contents(contents, instruction_name, input_name, response_name):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in content_type:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'excel' in content_type:
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'json' in content_type:
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        else:
            raise ValueError('Unsupported file format')

        df = rename_columns(df, instruction_name, input_name, response_name)
        return df
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def load_hf_dataset(path, instruction_name, input_name, response_name):
    try:
        ds = load_dataset(path)
        df = pd.DataFrame(ds['train'])
        df = rename_columns(df, instruction_name, input_name, response_name)
        return df
    
    except Exception as e:
        print(f"Error loading Hugging Face dataset: {str(e)}")
        return html.Div(f"Error loading Hugging Face dataset: {str(e)}")

def rename_columns(df, instruction_name=None, input_name=None, response_name=None):
    renaming_map = {}
    
    input_alternatives = ['input', 'prompt']
    response_alternatives = ['response', 'output']

    if instruction_name and instruction_name in df.columns:
        renaming_map[instruction_name] = 'instruction'
    
    if input_name and input_name in df.columns:
        if 'instruction' in renaming_map.values() or 'instruction' in df.columns:
            renaming_map[input_name] = 'input'
        else:
            renaming_map[input_name] = 'prompt'
    else:
        for name in input_alternatives:
            if name in df.columns:
                if 'instruction' in renaming_map.values() or 'instruction' in df.columns:
                    renaming_map[name] = 'input'
                else:
                    renaming_map[name] = 'prompt'
                break
    
    if response_name and response_name in df.columns:
        renaming_map[response_name] = 'response'
    else:
        for name in response_alternatives:
            if name in df.columns:
                renaming_map[name] = 'response'
                break
    
    df.rename(columns=renaming_map, inplace=True)
    
    if 'instruction' in df.columns and 'input' in df.columns:
        df['prompt'] = df['instruction'] + ". " + df['input']
    
    return df

# Global variables to store time estimates
average_time_per_row = None
time_estimates = []

# Callback to process the uploaded data
@app.callback(
    [
        Output('stored-dataframe', 'data'),
        Output('upload-status', 'children'),
        Output('dataset-loaded', 'data')
    ],
    [Input('process-button', 'n_clicks')],
    [
        State('upload-data', 'contents'),
        State('df-hf', 'value'),
        State('instruction-name', 'value'),
        State('input-name', 'value'),
        State('response-name', 'value'),
        State('manual-prompt', 'value'),  # Adicionado estado para o campo de entrada manual do Prompt
        State('manual-response', 'value')  # Adicionado estado para o campo de entrada manual do Response
    ]
)
def process_data(n_clicks, upload_contents, hf_path, instruction_name, input_name, response_name, manual_prompt, manual_response):
    if n_clicks == 0:
        return [dash.no_update, 'Upload or provide a dataset path.', dash.no_update]

    if manual_prompt and manual_response:
        df = pd.DataFrame({'prompt': [manual_prompt], 'response': [manual_response]})
        return [df.to_dict('records'), 'Manual input loaded as dataset.', True]

    if upload_contents:
        df = parse_contents(upload_contents, instruction_name, input_name, response_name)
    elif hf_path:
        df = load_hf_dataset(hf_path, instruction_name, input_name, response_name)
    else:
        return [dash.no_update, 'No dataset provided.', dash.no_update]

    if df is None:
        return [dash.no_update, 'Error loading dataset.', False]
    
    return [df.to_dict('records'), 'Dataset loaded successfully.', True]


# Callback para lidar com o clique do botão de análise de toxicidade e atualizar a barra de progresso
@app.callback(
    [
        Output('toxicity-results', 'data'),
        Output("progress", "value"),
        Output("progress", "label"),
        Output("progress-interval", "disabled"),
        Output("time-estimate", "children"),
        Output("progress", "style"),
        Output("toxicity-process-button", "style")
    ],
    [
        Input('toxicity-process-button', 'n_clicks'),
        Input('progress-interval', 'n_intervals')
    ],
    [
        State('stored-dataframe', 'data'),
        State('toxicity-results', 'data')
    ],
    prevent_initial_call=True
)
def process_toxicity_analysis(toxicity_n_clicks, n_intervals, data, current_results):
    global average_time_per_row
    global time_estimates

    df = pd.DataFrame(data)
    toxicity_analyzer = ToxicityAnalyzer()

    if ctx.triggered_id == 'toxicity-process-button':
        if toxicity_n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update
        
        print("Toxicity button clicked. Starting analysis...")
        time_estimates = []
        average_time_per_row = None

        initial_estimated_time = 0.1 * len(df) * 3
        return [], 0, "0%", False, f"Estimated time remaining: {initial_estimated_time:.2f} seconds", {"display": "block"}, {"display": "block"}

    if ctx.triggered_id != 'progress-interval':
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    if not data:
        print("No data available.")
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update
    
    if n_intervals <= len(df):
        start_time = time.time()
        row = df.iloc[n_intervals-1]
        print(f"Processing row: {n_intervals} of {len(df)}")
        row_result = toxicity_analyzer.analyze(pd.DataFrame([row]))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_estimates.append(elapsed_time)

        if len(time_estimates) == 1 and average_time_per_row is None:
            average_time_per_row = sum(time_estimates) / len(time_estimates)

        if current_results is None:
            current_results = []
        current_results.append(row_result.to_dict('records')[0])

        progress = ((n_intervals) / len(df)) * 100
        print(f"Progress: {progress:.2f}%")

        if average_time_per_row:
            n_df = len(df) - n_intervals 
            estimated_time_remaining = average_time_per_row * n_df 
            time_estimate_text = f"Estimated time remaining: {estimated_time_remaining:.2f} seconds"
        else:
            initial_estimated_time = 0.1 * len(df) * 2
            time_estimate_text = f"Estimated time remaining: {initial_estimated_time:.2f} seconds"

        return current_results, progress, f"{progress:.0f}%", False, time_estimate_text, {"display": "block"}, {"display": "block"}
    
    print("Processing complete.")
    return current_results, 100, "100%", True, "Processing complete.", {"display": "none"}, {"display": "none"}


# Callback para exibir os resultados de toxicidade
@app.callback(
    Output('toxicity-analysis-results', 'children'),
    Input('toxicity-results', 'data'),
    prevent_initial_call=True
)
def display_toxicity_results(toxicity_results):
    if toxicity_results:
        results_df = pd.DataFrame(toxicity_results)
        selected_columns = ['prompt', 'response', 'prompt_toxicity', 'response_toxicity']
        results_df = results_df[selected_columns]

        max_prompt_toxicity = results_df['prompt_toxicity'].max()
        mean_prompt_toxicity = results_df['prompt_toxicity'].mean()
        std_prompt_toxicity = results_df['prompt_toxicity'].std()

        max_response_toxicity = results_df['response_toxicity'].max()
        mean_response_toxicity = results_df['response_toxicity'].mean()
        std_response_toxicity = results_df['response_toxicity'].std()

        prompt_density_fig = px.histogram(
            results_df,
            x="prompt_toxicity",
            nbins=50,
            title='Density of Prompt Toxicity',
            marginal="violin",
            labels={"prompt_toxicity": "Prompt Toxicity"},
            color_discrete_sequence=["#636EFA"],
            template="plotly_white"
        )

        response_density_fig = px.histogram(
            results_df,
            x="response_toxicity",
            nbins=50,
            title='Density of Response Toxicity',
            marginal="violin",
            labels={"response_toxicity": "Response Toxicity"},
            color_discrete_sequence=["#636EFA"],
            template="plotly_white"
        )

        response_density_fig.update_xaxes(tickformat=".5f")

        table = dash_table.DataTable(
            id='interactive-table',
            columns=[
                {'name': 'Prompt', 'id': 'prompt'},
                {'name': 'Response', 'id': 'response'},
                {'name': 'Prompt Toxicity', 'id': 'prompt_toxicity'},
                {'name': 'Response Toxicity', 'id': 'response_toxicity'}
            ],
            data=results_df.to_dict('records'),
            page_size=3,
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_cell_conditional=[
                {'if': {'column_id': 'prompt'}, 'width': '30%'},
                {'if': {'column_id': 'response'}, 'width': '50%'},
                {'if': {'column_id': 'prompt_toxicity'}, 'width': '10%'},
                {'if': {'column_id': 'response_toxicity'}, 'width': '10%'},
            ],
            editable=True,
            export_format="csv",
            export_headers="display",
        )

        return dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Prompt Toxicity", className="card-title"),
                        html.P(f"{max_prompt_toxicity:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Mean Prompt Toxicity", className="card-title"),
                        html.P(f"{mean_prompt_toxicity:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Std Prompt Toxicity", className="card-title"),
                        html.P(f"{std_prompt_toxicity:.2f}", className="card-text")
                    ])
                ]), width=4)
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Response Toxicity", className="card-title"),
                        html.P(f"{max_response_toxicity:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Mean Response Toxicity", className="card-title"),
                        html.P(f"{mean_response_toxicity:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Std Response Toxicity", className="card-title"),
                        html.P(f"{std_response_toxicity:.2f}", className="card-text")
                    ])
                ]), width=4)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=prompt_density_fig), width=6),
                dbc.Col(dcc.Graph(figure=response_density_fig), width=6),
            ]),
            dbc.Row([
                dbc.Col(table, width=12)
            ])
        ])

    return dash.no_update


# Callback para lidar com o clique do botão de análise de sentimento e atualizar a barra de progresso
@app.callback(
    [
        Output('sentiment-results', 'data'),
        Output("sentiment-progress", "value"),
        Output("sentiment-progress", "label"),
        Output("sentiment-progress-interval", "disabled"),
        Output("sentiment-time-estimate", "children"),
        Output("sentiment-progress", "style"),
        Output("sentiment-process-button", "style")
    ],
    [
        Input('sentiment-process-button', 'n_clicks'),
        Input('sentiment-progress-interval', 'n_intervals')
    ],
    [
        State('stored-dataframe', 'data'),
        State('sentiment-results', 'data')
    ],
    prevent_initial_call=True
)
def process_sentiment_analysis(sentiment_n_clicks, n_intervals, data, current_results):
    global average_time_per_row
    global time_estimates

    df = pd.DataFrame(data)
    sentiment_analyzer = SentimentAnalyzer()

    if ctx.triggered_id == 'sentiment-process-button':
        if sentiment_n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update
        
        print("Sentiment button clicked. Starting analysis...")
        time_estimates = []
        average_time_per_row = None

        initial_estimated_time = 0.1 * len(df) * 3
        return [], 0, "0%", False, f"Estimated time remaining: {initial_estimated_time:.2f} seconds", {"display": "block"}, {"display": "block"}

    if ctx.triggered_id != 'sentiment-progress-interval':
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    if not data:
        print("No data available.")
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update
    

    if n_intervals <= len(df):
        start_time = time.time()
        row = df.iloc[n_intervals-1]
        print(f"Processing row: {n_intervals} of {len(df)}")
        row_result = sentiment_analyzer.analyze(pd.DataFrame([row]))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_estimates.append(elapsed_time)

        if len(time_estimates) == 1 and average_time_per_row is None:
            average_time_per_row = sum(time_estimates) / len(time_estimates)

        if current_results is None:
            current_results = []
        current_results.append(row_result.to_dict('records')[0])

        progress = ((n_intervals) / len(df)) * 100
        print(f"Progress: {progress:.2f}%")

        if average_time_per_row:
            n_df = len(df) - n_intervals 
            estimated_time_remaining = average_time_per_row * n_df * 2
            time_estimate_text = f"Estimated time remaining: {estimated_time_remaining:.2f} seconds"
        else:
            initial_estimated_time = 0.2 * len(df)
            time_estimate_text = f"Estimated time remaining: {initial_estimated_time:.2f} seconds"

        return current_results, progress, f"{progress:.0f}%", False, time_estimate_text, {"display": "block"}, {"display": "block"}
    
    print("Processing complete.")
    return current_results, 100, "100%", True, "Processing complete.", {"display": "none"}, {"display": "none"}


# Callback para exibir os resultados de sentimento
@app.callback(
    Output('sentiment-analysis-results', 'children'),
    Input('sentiment-results', 'data'),
    prevent_initial_call=True
)
def display_sentiment_results(sentiment_results):
    if sentiment_results:
        results_df = pd.DataFrame(sentiment_results)
        selected_columns = ['prompt', 'response', 'prompt_sentiment', 'response_sentiment']
        results_df = results_df[selected_columns]

        max_prompt_sentiment = results_df['prompt_sentiment'].max()
        mean_prompt_sentiment = results_df['prompt_sentiment'].mean()
        std_prompt_sentiment = results_df['prompt_sentiment'].std()

        max_response_sentiment = results_df['response_sentiment'].max()
        mean_response_sentiment = results_df['response_sentiment'].mean()
        std_response_sentiment = results_df['response_sentiment'].std()

        prompt_density_fig = px.histogram(
            results_df,
            x="prompt_sentiment",
            nbins=50,
            title='Density of Prompt Sentiment',
            marginal="violin",
            labels={"prompt_sentiment": "Prompt Sentiment"},
            color_discrete_sequence=["#636EFA"],
            template="plotly_white"
        )

        response_density_fig = px.histogram(
            results_df,
            x="response_sentiment",
            nbins=50,
            title='Density of Response Sentiment',
            marginal="violin",
            labels={"response_sentiment": "Response Sentiment"},
            color_discrete_sequence=["#636EFA"],
            template="plotly_white"
        )

        response_density_fig.update_xaxes(tickformat=".5f")

        table = dash_table.DataTable(
            id='interactive-table',
            columns=[
                {'name': 'Prompt', 'id': 'prompt'},
                {'name': 'Response', 'id': 'response'},
                {'name': 'Prompt Sentiment', 'id': 'prompt_sentiment'},
                {'name': 'Response Sentiment', 'id': 'response_sentiment'}
            ],
            data=results_df.to_dict('records'),
            page_size=3,
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_cell_conditional=[
                {'if': {'column_id': 'prompt'}, 'width': '30%'},
                {'if': {'column_id': 'response'}, 'width': '50%'},
                {'if': {'column_id': 'prompt_sentiment'}, 'width': '10%'},
                {'if': {'column_id': 'response_sentiment'}, 'width': '10%'},
            ],
            editable=True,
            export_format="csv",
            export_headers="display",
        )

        return dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Prompt Sentiment", className="card-title"),
                        html.P(f"{max_prompt_sentiment:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Mean Prompt Sentiment", className="card-title"),
                        html.P(f"{mean_prompt_sentiment:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Std Prompt Sentiment", className="card-title"),
                        html.P(f"{std_prompt_sentiment:.2f}", className="card-text")
                    ])
                ]), width=4)
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Response Sentiment", className="card-title"),
                        html.P(f"{max_response_sentiment:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Mean Response Sentiment", className="card-title"),
                        html.P(f"{mean_response_sentiment:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Std Response Sentiment", className="card-title"),
                        html.P(f"{std_response_sentiment:.2f}", className="card-text")
                    ])
                ]), width=4)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=prompt_density_fig), width=6),
                dbc.Col(dcc.Graph(figure=response_density_fig), width=6),
            ]),
            dbc.Row([
                dbc.Col(table, width=12)
            ])
        ])

    return dash.no_update

@app.callback(
    [
        Output('prompt-results', 'data'),
        Output("prompt-progress", "value"),
        Output("prompt-progress", "label"),
        Output("prompt-progress-interval", "disabled"),
        Output("prompt-time-estimate", "children"),
        Output("prompt-progress", "style"),
        Output("prompt-process-button", "style")
    ],
    [
        Input('prompt-process-button', 'n_clicks'),
        Input('prompt-progress-interval', 'n_intervals')
    ],
    [
        State('stored-dataframe', 'data'),
        State('prompt-results', 'data')
    ],
    prevent_initial_call=True
)
def process_prompt_analysis(prompt_n_clicks, n_intervals, data, current_results):
    global average_time_per_row
    global time_estimates

    df = pd.DataFrame(data)
    prompt_analyzer = PromptAnalyzer()

    if ctx.triggered_id == 'prompt-process-button':
        if prompt_n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update
        
        print("Prompt button clicked. Starting analysis...")
        time_estimates = []
        average_time_per_row = None

        initial_estimated_time = 0.1 * len(df) * 3
        return [], 0, "0%", False, f"Estimated time remaining: {initial_estimated_time:.2f} seconds", {"display": "block"}, {"display": "block"}

    if ctx.triggered_id != 'prompt-progress-interval':
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    if not data:
        print("No data available.")
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update
    
    
    if n_intervals <= len(df):
        start_time = time.time()
        row = df.iloc[n_intervals-1]
        print(f"Processing row: {n_intervals} of {len(df)}")
        row_result = prompt_analyzer.analyze(pd.DataFrame([row]))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_estimates.append(elapsed_time)

        if len(time_estimates) == 1 and average_time_per_row is None:
            average_time_per_row = sum(time_estimates) / len(time_estimates)

        if current_results is None:
            current_results = []
        current_results.append(row_result.to_dict('records')[0])

        progress = ((n_intervals) / len(df)) * 100
        print(f"Progress: {progress:.2f}%")

        if average_time_per_row:
            n_df = len(df) - n_intervals 
            estimated_time_remaining = average_time_per_row * n_df
            time_estimate_text = f"Estimated time remaining: {estimated_time_remaining:.2f} seconds"
        else:
            initial_estimated_time = 0.1 * len(df) * 2
            time_estimate_text = f"Estimated time remaining: {initial_estimated_time:.2f} seconds"

        return current_results, progress, f"{progress:.0f}%", False, time_estimate_text, {"display": "block"}, {"display": "block"}
    
    print("Processing complete.")
    return current_results, 100, "100%", True, "Processing complete.", {"display": "none"}, {"display": "none"}

@app.callback(
    Output('prompt-analysis-results', 'children'),
    Input('prompt-results', 'data'),
    prevent_initial_call=True
)
def display_prompt_results(prompt_results):
    if prompt_results:
        results_df = pd.DataFrame(prompt_results)
        selected_columns = ['prompt', 'prompt_injection', 'prompt_jailbreak']
        
        # Converte as listas para valores simples (sem colchetes)
        for col in selected_columns[1:]:  # Ignora a coluna 'prompt'
            results_df[col] = results_df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
        
        # Calcula max, mean, e std para cada coluna
        max_prompt_injection = results_df['prompt_injection'].max()
        mean_prompt_injection = results_df['prompt_injection'].mean()
        std_prompt_injection = results_df['prompt_injection'].std()


        max_prompt_jailbreak = results_df['prompt_jailbreak'].max()
        mean_prompt_jailbreak = results_df['prompt_jailbreak'].mean()
        std_prompt_jailbreak = results_df['prompt_jailbreak'].std()

        # Cria os gráficos de densidade
        prompt_injection_fig = px.histogram(
            results_df,
            x="prompt_injection",
            nbins=50,
            title='Density of Prompt Injection',
            marginal="violin",
            labels={"prompt_injection": "Prompt Injection"},
            color_discrete_sequence=["#636EFA"],
            template="plotly_white"
        )

        prompt_jailbreak_fig = px.histogram(
            results_df,
            x="prompt_jailbreak",
            nbins=50,
            title='Density of Prompt Jailbreak',
            marginal="violin",
            labels={"prompt_jailbreak": "Prompt Jailbreak"},
            color_discrete_sequence=["#636EFA"],
            template="plotly_white"
        )

        # Cria a tabela de dados
        table = dash_table.DataTable(
            id='interactive-table-prompt',
            columns=[
                {'name': 'Prompt', 'id': 'prompt'},
                {'name': 'Injection Score', 'id': 'prompt_injection'},
                {'name': 'Jailbreak Score', 'id': 'prompt_jailbreak'}
            ],
            data=results_df.to_dict('records'),
            page_size=3,
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            editable=True,
            export_format="csv",
            export_headers="display",
        )

        # Retorna a interface com os valores calculados e os gráficos
        return dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Prompt Injection", className="card-title"),
                        html.P(f"{max_prompt_injection:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Mean Prompt Injection", className="card-title"),
                        html.P(f"{mean_prompt_injection:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Std Prompt Injection", className="card-title"),
                        html.P(f"{std_prompt_injection:.2f}", className="card-text")
                    ])
                ]), width=4)
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Prompt Jailbreak", className="card-title"),
                        html.P(f"{max_prompt_jailbreak:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Mean Prompt Jailbreak", className="card-title"),
                        html.P(f"{mean_prompt_jailbreak:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Std Prompt Jailbreak", className="card-title"),
                        html.P(f"{std_prompt_jailbreak:.2f}", className="card-text")
                    ])
                ]), width=4)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=prompt_injection_fig), width=6),
                dbc.Col(dcc.Graph(figure=prompt_jailbreak_fig), width=6),
            ]),
            dbc.Row([
                dbc.Col(table, width=12)
            ])
        ])

    return dash.no_update

@app.callback(
    [
        Output('refusal-results', 'data'),
        Output("refusal-progress", "value"),
        Output("refusal-progress", "label"),
        Output("refusal-progress-interval", "disabled"),
        Output("refusal-time-estimate", "children"),
        Output("refusal-progress", "style"),
        Output("refusal-process-button", "style")
    ],
    [
        Input('refusal-process-button', 'n_clicks'),
        Input('refusal-progress-interval', 'n_intervals')
    ],
    [
        State('stored-dataframe', 'data'),
        State('refusal-results', 'data')
    ],
    prevent_initial_call=True
)
def process_refusal_analysis(refusal_n_clicks, n_intervals, data, current_results):
    global average_time_per_row
    global time_estimates

    df = pd.DataFrame(data)
    refusal_analyzer = RefusalAnalyzer()

    if ctx.triggered_id == 'refusal-process-button':
        if refusal_n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update
        
        print("Refusal button clicked. Starting analysis...")
        time_estimates = []
        average_time_per_row = None

        initial_estimated_time = 0.1 * len(df) * 3
        return [], 0, "0%", False, f"Estimated time remaining: {initial_estimated_time:.2f} seconds", {"display": "block"}, {"display": "block"}

    if ctx.triggered_id != 'refusal-progress-interval':
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    if not data:
        print("No data available.")
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update


    if n_intervals <= len(df):
        start_time = time.time()
        row = df.iloc[n_intervals-1]
        print(f"Processing row: {n_intervals + 1} of {len(df)}")
        row_result = refusal_analyzer.analyze(pd.DataFrame([row]))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_estimates.append(elapsed_time)

        if len(time_estimates) == 1 and average_time_per_row is None:
            average_time_per_row = sum(time_estimates) / len(time_estimates)

        if current_results is None:
            current_results = []
        current_results.append(row_result.to_dict('records')[0])

        progress = ((n_intervals) / len(df)) * 100
        print(f"Progress: {progress:.2f}%")

        if average_time_per_row:
            n_df = len(df) - n_intervals 
            estimated_time_remaining = average_time_per_row * n_df * 2
            time_estimate_text = f"Estimated time remaining: {estimated_time_remaining:.2f} seconds"
        else:
            initial_estimated_time = 0.1 * len(df) * 2
            time_estimate_text = f"Estimated time remaining: {initial_estimated_time:.2f} seconds"

        return current_results, progress, f"{progress:.0f}%", False, time_estimate_text, {"display": "block"}, {"display": "block"}

    print("Processing complete.")
    return current_results, 100, "100%", True, "Processing complete.", {"display": "none"}, {"display": "none"}

@app.callback(
    Output('refusal-analysis-results', 'children'),
    Input('refusal-results', 'data'),
    prevent_initial_call=True
)
def display_refusal_results(refusal_results):
    if refusal_results:
        results_df = pd.DataFrame(refusal_results)
        selected_columns = ['prompt', 'response', 'refusal']
        results_df = results_df[selected_columns]

        max_refusal = results_df['refusal'].max()
        mean_refusal = results_df['refusal'].mean()
        std_refusal = results_df['refusal'].std()

        # Cria os gráficos de densidade
        refusal_fig = px.histogram(
            results_df,
            x="refusal",
            nbins=50,
            title='Density of Prompt Injection',
            marginal="violin",
            labels={"prompt_injection": "Prompt Injection"},
            color_discrete_sequence=["#636EFA"],
            template="plotly_white"
        )

        # Exibir uma tabela com os resultados de refusal
        table = dash_table.DataTable(
            id='interactive-table-refusal',
            columns=[
                {'name': 'Prompt', 'id': 'prompt'},
                {'name': 'Response', 'id': 'response'},
                {'name': 'Refusal Score', 'id': 'refusal'}
            ],
            data=results_df.to_dict('records'),
            page_size=3,
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            editable=True,
            export_format="csv",
            export_headers="display",
        )

        return dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Max Refusal", className="card-title"),
                        html.P(f"{max_refusal:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Mean Refusal", className="card-title"),
                        html.P(f"{mean_refusal:.2f}", className="card-text")
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Std Refusal", className="card-title"),
                        html.P(f"{std_refusal:.2f}", className="card-text")
                    ])
                ]), width=4)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=refusal_fig), width=12)
            ]),
            dbc.Row([
                dbc.Col(table, width=12)
            ])
        ])

    return dash.no_update

@app.callback(
    [
        Output('topics-results', 'data'),
        Output("topics-progress", "value"),
        Output("topics-progress", "label"),
        Output("topics-progress-interval", "disabled"),
        Output("topics-time-estimate", "children"),
        Output("topics-progress", "style"),
        Output("topics-process-button", "style")
    ],
    [
        Input('topics-process-button', 'n_clicks'),
        Input('topics-progress-interval', 'n_intervals')
    ],
    [
        State('stored-dataframe', 'data'),
        State('topics-input', 'value'),  # Adiciona o estado para capturar os tópicos inseridos pelo usuário
        State('topics-results', 'data')
    ],
    prevent_initial_call=True
)
def process_topics_analysis(topics_n_clicks, n_intervals, data, topics_input, current_results):
    global average_time_per_row
    global time_estimates

    if not data:
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    df = pd.DataFrame(data)
    topics_analyzer = TopicsAnalyzer()

    # Processa a lista de tópicos inseridos pelo usuário
    topics_list = [topic.strip() for topic in topics_input.split(',')] if topics_input else None

    if current_results is None:
        current_results = []

    if ctx.triggered_id == 'topics-process-button':
        if topics_n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

        # Reset time estimates and progress
        time_estimates = []
        average_time_per_row = None

        initial_estimated_time = 0.1 * len(df) * 3
        return [], 0, "0%", False, f"Estimated time remaining: {initial_estimated_time:.2f} seconds", {"display": "block"}, {"display": "block"}

    if ctx.triggered_id == 'topics-progress-interval':
        if n_intervals <= len(df):
            start_time = time.time()

            row = df.iloc[n_intervals-1]
            print(f"Processing row: {n_intervals} of {len(df)}")
            row_result = topics_analyzer.analyze(pd.DataFrame([row]), topics_list=topics_list)

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_estimates.append(elapsed_time)

            if len(time_estimates) > 0 and average_time_per_row is None:
                average_time_per_row = sum(time_estimates) / len(time_estimates)

            current_results.append(row_result.to_dict('records')[0])

            progress = ((n_intervals) / len(df)) * 100
            print(f"Progress: {progress:.2f}%")

            if average_time_per_row:
                n_df = len(df) - n_intervals 
                estimated_time_remaining = average_time_per_row * n_df * 2
                time_estimate_text = f"Estimated time remaining: {estimated_time_remaining:.2f} seconds"
            else:
                initial_estimated_time = 0.1 * len(df) * 2
                time_estimate_text = f"Estimated time remaining: {initial_estimated_time:.2f} seconds"

            return current_results, progress, f"{progress:.0f}%", False, time_estimate_text, {"display": "block"}, {"display": "block"}

        else:
            print("Processing complete.")
            return current_results, 100, "100%", True, "Processing complete.", {"display": "none"}, {"display": "none"}

    return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output('topics-analysis-results', 'children'),
    Input('topics-results', 'data'),
    prevent_initial_call=True
)
def display_topics_results(topics_results):
    if topics_results:
        results_df = pd.DataFrame(topics_results)
        selected_columns = ['prompt', 'response', 'prompt_topics', 'response_topics']
        results_df = results_df[selected_columns]

        # Calcula a frequência dos tópicos para prompt e response
        prompt_topic_counts = Counter(results_df['prompt_topics'])
        response_topic_counts = Counter(results_df['response_topics'])

        # Ordena os tópicos em ordem alfabética
        sorted_prompt_topics = sorted(prompt_topic_counts.keys())
        sorted_response_topics = sorted(response_topic_counts.keys())

        # Obtém os tópicos mais frequentes
        top_prompt_topic = prompt_topic_counts.most_common(1)[0][0]
        top_response_topic = response_topic_counts.most_common(1)[0][0]

        # Cria gráficos de barra para a distribuição de tópicos
        prompt_topic_fig = px.bar(
            x=sorted_prompt_topics,
            y=[prompt_topic_counts[topic] for topic in sorted_prompt_topics],
            title="Distribution of Prompt Topics",
            labels={'x': 'Topics', 'y': 'Count'},
            template="plotly_white"
        )

        response_topic_fig = px.bar(
            x=sorted_response_topics,
            y=[response_topic_counts[topic] for topic in sorted_response_topics],
            title="Distribution of Response Topics",
            labels={'x': 'Topics', 'y': 'Count'},
            template="plotly_white"
        )

        # Tabela para exibir os resultados de tópicos
        table = dash_table.DataTable(
            id='interactive-table-topics',
            columns=[
                {'name': 'Prompt', 'id': 'prompt'},
                {'name': 'Response', 'id': 'response'},
                {'name': 'Prompt Topics', 'id': 'prompt_topics'},
                {'name': 'Response Topics', 'id': 'response_topics'}
            ],
            data=results_df.to_dict('records'),
            page_size=3,
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            editable=True,
            export_format="csv",
            export_headers="display",
        )

        return dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Most Common Prompt Topic", className="card-title"),
                        html.P(top_prompt_topic, className="card-text")
                    ])
                ]), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Most Common Response Topic", className="card-title"),
                        html.P(top_response_topic, className="card-text")
                    ])
                ]), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=prompt_topic_fig), width=6),
                dbc.Col(dcc.Graph(figure=response_topic_fig), width=6),
            ]),
            dbc.Row([
                dbc.Col(table, width=12)
            ])
        ])

    return dash.no_update

@app.callback(
    [
        Output('patterns-results', 'data'),  # Certifique-se de que o ID seja 'patterns-results'
        Output("patterns-progress", "value"),
        Output("patterns-progress", "label"),
        Output("patterns-progress-interval", "disabled"),
        Output("patterns-time-estimate", "children"),
        Output("patterns-progress", "style"),
        Output("patterns-process-button", "style")
    ],
    [
        Input('patterns-process-button', 'n_clicks'),
        Input('patterns-progress-interval', 'n_intervals')
    ],
    [
        State('stored-dataframe', 'data'),
        State('patterns-results', 'data')  # Certifique-se de que o ID seja 'patterns-results'
    ],
    prevent_initial_call=True
)

def process_patterns_analysis(patterns_n_clicks, n_intervals, data, current_results):
    global average_time_per_row
    global time_estimates

    df = pd.DataFrame(data)
    patterns_analyzer = RegexAnalyzer()

    if ctx.triggered_id == 'patterns-process-button':
        if patterns_n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update
        
        print("Patterns button clicked. Starting analysis...")
        time_estimates = []
        average_time_per_row = None

        initial_estimated_time = 0.1 * len(df) * 3
        return [], 0, "0%", False, f"Estimated time remaining: {initial_estimated_time:.2f} seconds", {"display": "block"}, {"display": "block"}

    if ctx.triggered_id != 'patterns-progress-interval':
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    if not data:
        print("No data available.")
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update
    
    if n_intervals <= len(df):
        start_time = time.time()
        row = df.iloc[n_intervals-1]
        print(f"Processing row: {n_intervals} of {len(df)}")
        row_result = patterns_analyzer.analyze(pd.DataFrame([row]))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_estimates.append(elapsed_time)

        if len(time_estimates) == 1 and average_time_per_row is None:
            average_time_per_row = sum(time_estimates) / len(time_estimates)

        if current_results is None:
            current_results = []
        current_results.append(row_result.to_dict('records')[0])

        progress = ((n_intervals) / len(df)) * 100
        print(f"Progress: {progress:.2f}%")

        if average_time_per_row:
            n_df = len(df) - n_intervals 
            estimated_time_remaining = average_time_per_row * n_df
            time_estimate_text = f"Estimated time remaining: {estimated_time_remaining:.2f} seconds"
        else:
            initial_estimated_time = 0.1 * len(df) * 2
            time_estimate_text = f"Estimated time remaining: {initial_estimated_time:.2f} seconds"

        return current_results, progress, f"{progress:.0f}%", False, time_estimate_text, {"display": "block"}, {"display": "block"}
    
    print("Processing complete.")
    return current_results, 100, "100%", True, "Processing complete.", {"display": "none"}, {"display": "none"}

@app.callback(
    Output('patterns-analysis-results', 'children'),
    Input('patterns-results', 'data'),
    prevent_initial_call=True
)
def display_patterns_results(patterns_results):
    if patterns_results:
        results_df = pd.DataFrame(patterns_results)
        selected_columns = ['prompt', 'response', 'prompt_patterns', 'response_patterns']
        results_df = results_df[selected_columns]

        # Calcula a frequência dos padrões para prompt e response, ignorando None
        prompt_patterns_counts = Counter(results_df['prompt_patterns'])
        response_patterns_counts = Counter(results_df['response_patterns'])

        # Filtra o padrão mais comum ignorando 'None'
        top_prompt_patterns = [pattern for pattern in prompt_patterns_counts if pattern is not None]
        top_prompt_patterns = top_prompt_patterns[0] if top_prompt_patterns else "None"

        top_response_patterns = [pattern for pattern in response_patterns_counts if pattern is not None]
        top_response_patterns = top_response_patterns[0] if top_response_patterns else "None"

        # Cria gráficos de barra para a distribuição de padrões
        prompt_patterns_fig = px.bar(
            x=list(prompt_patterns_counts.keys()),  # Passa as chaves do contador como lista para x
            y=list(prompt_patterns_counts.values()),  # Passa os valores do contador como lista para y
            title="Distribution of Prompt Patterns",
            labels={'x': 'Patterns', 'y': 'Count'},
            template="plotly_white"
        )

        response_patterns_fig = px.bar(
            x=list(response_patterns_counts.keys()),  # Passa as chaves do contador como lista para x
            y=list(response_patterns_counts.values()),  # Passa os valores do contador como lista para y
            title="Distribution of Response Patterns",
            labels={'x': 'Patterns', 'y': 'Count'},
            template="plotly_white"
        )

        # Tabela para exibir os resultados de padrões
        table = dash_table.DataTable(
            id='interactive-table-patterns',
            columns=[
                {'name': 'Prompt', 'id': 'prompt'},
                {'name': 'Response', 'id': 'response'},
                {'name': 'Prompt Patterns', 'id': 'prompt_patterns'},
                {'name': 'Response Patterns', 'id': 'response_patterns'}
            ],
            data=results_df.to_dict('records'),
            page_size=3,
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            editable=True,
            export_format="csv",
            export_headers="display",
        )

        return dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Most Common Prompt Patterns", className="card-title"),
                        html.P(top_prompt_patterns, className="card-text")
                    ])
                ]), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Most Common Response Patterns", className="card-title"),
                        html.P(top_response_patterns, className="card-text")
                    ])
                ]), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=prompt_patterns_fig), width=6),
                dbc.Col(dcc.Graph(figure=response_patterns_fig), width=6),
            ]),
            dbc.Row([
                dbc.Col(table, width=12)
            ])
        ])

    return dash.no_update

@app.callback(
    [
        Output('entity-results', 'data'),
        Output("entity-progress", "value"),
        Output("entity-progress", "label"),
        Output("entity-progress-interval", "disabled"),
        Output("entity-time-estimate", "children"),
        Output("entity-progress", "style"),
        Output("entity-process-button", "style")
    ],
    [
        Input('entity-process-button', 'n_clicks'),
        Input('entity-progress-interval', 'n_intervals')
    ],
    [
        State('stored-dataframe', 'data'),
        State('identificação_pessoal-selection', 'value'),  # Corrigido
        State('cartões_de_crédito-selection', 'value'),  # Corrigido
        State('localização-selection', 'value'),  # Corrigido
        State('documentos-selection', 'value'),  # Corrigido
        State('outros-selection', 'value'),  # Corrigido
        State('entity-results', 'data')
    ],
    prevent_initial_call=True
)
def process_entity_recognition(entity_n_clicks, n_intervals, data, identification_entities, credit_cards_entities, location_entities, documents_entities, others_entities, current_results):
    global average_time_per_row
    global time_estimates

    if not data:
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    df = pd.DataFrame(data)
    entity_analyzer = PIIAnalyzer()

    selected_entities = identification_entities + credit_cards_entities + location_entities + documents_entities + others_entities

    if current_results is None:
        current_results = []

    if ctx.triggered_id == 'entity-process-button':
        if entity_n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

        # Reset time estimates and progress
        time_estimates = []
        average_time_per_row = None

        initial_estimated_time = 0.1 * len(df) * 3
        return [], 0, "0%", False, f"Estimated time remaining: {initial_estimated_time:.2f} seconds", {"display": "block"}, {"display": "block"}

    if ctx.triggered_id == 'entity-progress-interval':
        if n_intervals <= len(df):
            start_time = time.time()

            row = df.iloc[n_intervals-1]
            print(f"Processing row: {n_intervals} of {len(df)}")
            row_result = entity_analyzer.analyze(pd.DataFrame([row]), selected_entities)

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_estimates.append(elapsed_time)

            if len(time_estimates) > 0 and average_time_per_row is None:
                average_time_per_row = sum(time_estimates) / len(time_estimates)

            current_results.append(row_result.to_dict('records')[0])

            progress = ((n_intervals) / len(df)) * 100
            print(f"Progress: {progress:.2f}%")

            if average_time_per_row:
                n_df = len(df) - n_intervals
                estimated_time_remaining = average_time_per_row * n_df * 2
                time_estimate_text = f"Estimated time remaining: {estimated_time_remaining:.2f} seconds"
            else:
                initial_estimated_time = 0.1 * len(df) * 2
                time_estimate_text = f"Estimated time remaining: {initial_estimated_time:.2f} seconds"

            return current_results, progress, f"{progress:.0f}%", False, time_estimate_text, {"display": "block"}, {"display": "block"}

        else:
            print("Processing complete.")
            return current_results, 100, "100%", True, "Processing complete.", {"display": "none"}, {"display": "none"}

    return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('entity-analysis-results', 'children'),
    Input('entity-results', 'data'),
    prevent_initial_call=True
)
def display_entity_results(entity_results):
    if entity_results:
        # Converte os resultados em um DataFrame
        results_df = pd.DataFrame(entity_results)

        # Inicializa contadores para entidades
        prompt_entities_counts = Counter()
        response_entities_counts = Counter()

        # Extrai e conta as entidades em 'prompt_pii' e 'response_pii'
        for _, row in results_df.iterrows():
            if row['prompt_pii']:
                for entity_info in row['prompt_pii']:
                    prompt_entities_counts[entity_info['Entidade']] += 1
            if row['response_pii']:
                for entity_info in row['response_pii']:
                    response_entities_counts[entity_info['Entidade']] += 1

        # Função para encontrar as entidades mais comuns, considerando empates
        def find_most_common_entities(entities_counter):
            if not entities_counter:
                return "None"
            # Frequência máxima
            max_count = max(entities_counter.values())
            # Entidades com frequência máxima
            most_common_entities = [entity for entity, count in entities_counter.items() if count == max_count]
            return ', '.join(most_common_entities)

        # Obtém as entidades mais comuns
        top_prompt_entities = find_most_common_entities(prompt_entities_counts)
        top_response_entities = find_most_common_entities(response_entities_counts)

        # Cria gráficos de barra para a distribuição de entidades
        if prompt_entities_counts:
            prompt_entity_fig = px.bar(
                x=list(prompt_entities_counts.keys()),
                y=list(prompt_entities_counts.values()),
                title="Distribution of Prompt Entities",
                labels={'x': 'Entities', 'y': 'Count'},
                template="plotly_white"
            )
        else:
            prompt_entity_fig = px.bar(
                x=["No Entities"],
                y=[0],
                title="Distribution of Prompt Entities",
                labels={'x': 'Entities', 'y': 'Count'},
                template="plotly_white"
            )

        if response_entities_counts:
            response_entity_fig = px.bar(
                x=list(response_entities_counts.keys()),
                y=list(response_entities_counts.values()),
                title="Distribution of Response Entities",
                labels={'x': 'Entities', 'y': 'Count'},
                template="plotly_white"
            )
        else:
            response_entity_fig = px.bar(
                x=["No Entities"],
                y=[0],
                title="Distribution of Response Entities",
                labels={'x': 'Entities', 'y': 'Count'},
                template="plotly_white"
            )

        # Converte as listas de dicionários em strings para o DataTable
        results_df['prompt_pii'] = results_df['prompt_pii'].apply(lambda x: ', '.join([f"{e['Entidade']}: {e['Texto']}" for e in x]))
        results_df['response_pii'] = results_df['response_pii'].apply(lambda x: ', '.join([f"{e['Entidade']}: {e['Texto']}" for e in x]))

        # Tabela para exibir os resultados de entidades
        table = dash_table.DataTable(
            id='interactive-table-entities',
            columns=[
                {'name': 'Prompt', 'id': 'prompt'},
                {'name': 'Response', 'id': 'response'},
                {'name': 'Prompt Entities', 'id': 'prompt_pii'},
                {'name': 'Response Entities', 'id': 'response_pii'}
            ],
            data=results_df.to_dict('records'),
            page_size=3,
            sort_action="native",
            filter_action="native",
            row_selectable="multi",
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            editable=True,
            export_format="csv",
            export_headers="display",
        )

        return dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Most Common Prompt Entities", className="card-title"),
                        html.P(top_prompt_entities, className="card-text")
                    ])
                ]), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H5("Most Common Response Entities", className="card-title"),
                        html.P(top_response_entities, className="card-text")
                    ])
                ]), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=prompt_entity_fig), width=6),
                dbc.Col(dcc.Graph(figure=response_entity_fig), width=6),
            ]),
            dbc.Row([
                dbc.Col(table, width=12)
            ])
        ])

    return html.Div("")


# Callback para processar a análise de TextStat
@app.callback(
    [
        Output('textstat-results', 'data'),
        Output("textstat-progress", "value"),
        Output("textstat-progress", "label"),
        Output("textstat-progress-interval", "disabled"),
        Output("textstat-time-estimate", "children"),
        Output("textstat-progress", "style"),
        Output("textstat-process-button", "style")
    ],
    [
        Input('textstat-process-button', 'n_clicks'),
        Input('textstat-progress-interval', 'n_intervals')
    ],
    [
        State('stored-dataframe', 'data'),
        State('textstat-results', 'data')
    ],
    prevent_initial_call=True
)
def process_textstat_analysis(textstat_n_clicks, n_intervals, data, current_results):
    global average_time_per_row
    global time_estimates

    df = pd.DataFrame(data)
    textstat_analyzer = TextStatAnalyzer(language='en')

    if ctx.triggered_id == 'textstat-process-button':
        if textstat_n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update
        
        print("TextStat button clicked. Starting analysis...")
        time_estimates = []
        average_time_per_row = None

        initial_estimated_time = 0.1 * len(df) * 3
        return [], 0, "0%", False, f"Estimated time remaining: {initial_estimated_time:.2f} seconds", {"display": "block"}, {"display": "block"}

    if ctx.triggered_id != 'textstat-progress-interval':
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update, dash.no_update

    if not data:
        print("No data available.")
        return dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, dash.no_update

    if n_intervals <= len(df):
        start_time = time.time()
        row = df.iloc[n_intervals-1]
        print(f"Processing row: {n_intervals} of {len(df)}")
        row_result = textstat_analyzer.analyze(pd.DataFrame([row]))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        time_estimates.append(elapsed_time)

        if len(time_estimates) == 1 and average_time_per_row is None:
            average_time_per_row = sum(time_estimates) / len(time_estimates)

        if current_results is None:
            current_results = []
        current_results.append(row_result.to_dict('records')[0])

        progress = ((n_intervals) / len(df)) * 100
        print(f"Progress: {progress:.2f}%")

        if average_time_per_row:
            n_df = len(df) - n_intervals 
            estimated_time_remaining = average_time_per_row * n_df 
            time_estimate_text = f"Estimated time remaining: {estimated_time_remaining:.2f} seconds"
        else:
            initial_estimated_time = 0.1 * len(df) * 2
            time_estimate_text = f"Estimated time remaining: {initial_estimated_time:.2f} seconds"

        return current_results, progress, f"{progress:.0f}%", False, time_estimate_text, {"display": "block"}, {"display": "block"}

    print("Processing complete.")
    return current_results, 100, "100%", True, "Processing complete.", {"display": "none"}, {"display": "none"}


# Callback para exibir os resultados de TextStat
# Modifique o callback display_textstat_results para combinar as métricas selecionadas
@app.callback(
    Output('textstat-analysis-results', 'children'),
    [
        Input('textstat-results', 'data'),
        Input('textstat-metric-checklist-col1', 'value'),
        Input('textstat-metric-checklist-col2', 'value'),
        Input('textstat-metric-checklist-col3', 'value')
    ],  # Adicione os checklists como input
    prevent_initial_call=True
)
def display_textstat_results(textstat_results, selected_metrics_col1, selected_metrics_col2, selected_metrics_col3):
    selected_metrics = selected_metrics_col1 + selected_metrics_col2 + selected_metrics_col3  # Combine todas as seleções

    if textstat_results and selected_metrics:
        results_df = pd.DataFrame(textstat_results)

        rows = []

        # Criar gráficos para cada métrica selecionada
        for metric in selected_metrics:
            prompt_col = f'Prompt {metric}'
            response_col = f'Response {metric}'

            if prompt_col in results_df.columns and response_col in results_df.columns:
                # Gráfico para Prompt
                fig_prompt = px.histogram(
                    results_df,
                    x=prompt_col,
                    nbins=50,
                    title=f'Distribution of {metric} (Prompt)',
                    labels={prompt_col: metric},
                    template="plotly_white"
                )

                # Gráfico para Response
                fig_response = px.histogram(
                    results_df,
                    x=response_col,
                    nbins=50,
                    title=f'Distribution of {metric} (Response)',
                    labels={response_col: metric},
                    template="plotly_white"
                )

                # Adicionar gráficos lado a lado
                row = dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_prompt), width=6),
                    dbc.Col(dcc.Graph(figure=fig_response), width=6)
                ])
                rows.append(row)

        return dbc.Container(rows)

    return html.Div("")


# Callback to switch between tabs
@app.callback(
    Output('main-content', 'children'),
    Input('link-data-view', 'n_clicks'),
    Input('link-sentiment', 'n_clicks'),
    Input('link-toxicity', 'n_clicks'),
    Input('link-prompt', 'n_clicks'),
    Input('link-refusal', 'n_clicks'),
    Input('link-topics', 'n_clicks'), 
    Input('link-patterns', 'n_clicks'),
    Input('link-entity-recognition', 'n_clicks'),
    Input('link-textstat', 'n_clicks'),  # Add this line
    State('stored-dataframe', 'data'),
    State('toxicity-results', 'data'),
    State('sentiment-results', 'data'),
    State('prompt-results', 'data'),
    State('refusal-results', 'data'),
    State('topics-results', 'data'),  
    State('patterns-results', 'data'),
    State('entity-results', 'data'),
    State('textstat-results', 'data'), 
    prevent_initial_call=True
)
def update_tab(n_clicks_data, n_clicks_sentiment, n_clicks_toxicity, n_clicks_prompt, n_clicks_refusal, n_clicks_topics, n_clicks_patterns, n_clicks_entity, n_clicks_textstat, stored_data, toxicity_results, sentiment_results, prompt_results, refusal_results, topics_results, patterns_results, entity_results, textstat_results):
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered_id

    if button_id == 'link-data-view':
        if stored_data:
            df = pd.DataFrame(stored_data)
            table = dash_table.DataTable(
                id='dataset-table',
                columns=[{'name': col, 'id': col} for col in df.columns],
                data=df.to_dict('records'),
                page_size=10,
                sort_action="native",
                filter_action="native",
                row_selectable="multi",
                selected_rows=[],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_header={
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                editable=True,
                export_format="csv",
                export_headers="display",
            )
            return html.Div([
                html.H3('Dataset View'),
                table
            ])
        else:
            return html.Div([
                html.H3('Dataset View'),
                html.P("No dataset loaded yet.")
            ])

    elif button_id == 'link-sentiment':
        if sentiment_results:
            return display_sentiment_results(sentiment_results)
        else:
            return html.Div([
                html.H3('Sentiment Analysis'),
                html.Button('Process Sentiment Analysis', id='sentiment-process-button', n_clicks=0),
                dbc.Progress(id="sentiment-progress", striped=True, animated=True, style={"marginTop": "20px"}),
                html.Div(id="sentiment-time-estimate", style={'marginTop': '10px', 'fontSize': '14px'}),
                dcc.Loading(id="sentiment-loading", type="circle", children=[
                    html.Div(id='sentiment-analysis-results'),
                ])
            ])

    elif button_id == 'link-toxicity':
        if toxicity_results:
            return display_toxicity_results(toxicity_results)
        else:
            return html.Div([
                html.H3('Toxicity Analysis'),
                html.Button('Process Toxicity Analysis', id='toxicity-process-button', n_clicks=0),
                dbc.Progress(id="progress", striped=True, animated=True, style={"marginTop": "20px"}),
                html.Div(id="time-estimate", style={'marginTop': '10px', 'fontSize': '14px'}),
                dcc.Loading(id="toxicity-loading", type="circle", children=[
                    html.Div(id='toxicity-analysis-results'),
                ])
            ])

    elif button_id == 'link-prompt':
        if prompt_results:
            return display_prompt_results(prompt_results)
        else:
            return html.Div([
                html.H3('Prompt Jailbreak and Injection'),
                html.Button('Process Prompt Analysis', id='prompt-process-button', n_clicks=0),
                dbc.Progress(id="prompt-progress", striped=True, animated=True, style={"marginTop": "20px"}),
                html.Div(id="prompt-time-estimate", style={'marginTop': '10px', 'fontSize': '14px'}),
                dcc.Loading(id="prompt-loading", type="circle", children=[
                    html.Div(id='prompt-analysis-results'),
                ])
            ])
    
    elif button_id == 'link-refusal':
        if refusal_results:
            return display_refusal_results(refusal_results)
        else:
            return html.Div([
                html.H3('Refusal Analysis'),
                html.Button('Process Refusal Analysis', id='refusal-process-button', n_clicks=0),
                dbc.Progress(id="refusal-progress", striped=True, animated=True, style={"marginTop": "20px"}),
                html.Div(id="refusal-time-estimate", style={'marginTop': '10px', 'fontSize': '14px'}),
                dcc.Loading(id="refusal-loading", type="circle", children=[
                    html.Div(id='refusal-analysis-results'),
                ])
            ])
    
    elif button_id == 'link-topics':
        if topics_results:
            return display_topics_results(topics_results)
        else:
            return html.Div([
                html.H3('Topics Analysis'),
                dcc.Input(
                    id='topics-input',
                    type='text',
                    placeholder='Enter topics separated by commas or leave blank for free analysis',
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                html.Button('Process Topics Analysis', id='topics-process-button', n_clicks=0),
                dbc.Progress(id="topics-progress", striped=True, animated=True, style={"marginTop": "20px"}),
                html.Div(id="topics-time-estimate", style={'marginTop': '10px', 'fontSize': '14px'}),
                dcc.Loading(id="topics-loading", type="circle", children=[
                    html.Div(id='topics-analysis-results'),
                ])
            ])
    
    elif button_id == 'link-patterns':
        if patterns_results:
            return display_patterns_results(patterns_results)
        else:
            return html.Div([
                html.H3('Pattern Analysis'),
                html.Button('Process Pattern Analysis', id='patterns-process-button', n_clicks=0),
                dbc.Progress(id="patterns-progress", striped=True, animated=True, style={"marginTop": "20px"}),
                html.Div(id="patterns-time-estimate", style={'marginTop': '10px', 'fontSize': '14px'}),
                dcc.Loading(id="patterns-loading", type="circle", children=[
                    html.Div(id='patterns-analysis-results'),
                ])
            ])
    
    elif button_id == 'link-entity-recognition':
        supported_entities = {
            'Documents': ["US_DRIVER_LICENSE", "AU_ABN", "AU_ACN", "AU_TFN", "IN_PAN", "IN_VEHICLE_REGISTRATION"],
            'Others': ["PHONE_NUMBER", "DATE_TIME", "MEDICAL_LICENSE", "EMAIL_ADDRESS", "ORGANIZATION", "URL", "CRYPTO", "UK_NHS", "BAN_CODE"],
            'Credit Cards': ["CREDIT_CARD", "US_ITIN", "US_BANK_NUMBER", "AU_TFN", "IBAN_CODE"],
            'Personal Identification': ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER'],
            'Location': ['LOCATION', 'IP_ADDRESS']
        }


        # Cards organizados por coluna específica
        column1 = [
            dbc.Card(
                dbc.CardBody([
                    html.H5('Credit Cards', className="card-title"),
                    dcc.Checklist(
                        id='cartões_de_crédito-selection',
                        options=[{'label': entity, 'value': entity} for entity in sorted(supported_entities['Credit Cards'])],
                        value=sorted(supported_entities['Credit Cards']),
                        style={'marginBottom': '10px'}
                    )
                ]),
                style={'marginBottom': '10px'}
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5('Personal Identification', className="card-title"),
                    dcc.Checklist(
                        id='identificação_pessoal-selection',
                        options=[{'label': entity, 'value': entity} for entity in sorted(supported_entities['Personal Identification'])],
                        value=sorted(supported_entities['Personal Identification']),
                        style={'marginBottom': '10px'}
                    )
                ]),
                style={'marginBottom': '10px'}
            )
        ]

        column2 = [
            dbc.Card(
                dbc.CardBody([
                    html.H5('Documents', className="card-title"),
                    dcc.Checklist(
                        id='documentos-selection',
                        options=[{'label': entity, 'value': entity} for entity in sorted(supported_entities['Documents'])],
                        value=sorted(supported_entities['Documents']),
                        style={'marginBottom': '10px'}
                    )
                ]),
                style={'marginBottom': '10px'}
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5('Location', className="card-title"),
                    dcc.Checklist(
                        id='localização-selection',
                        options=[{'label': entity, 'value': entity} for entity in sorted(supported_entities['Location'])],
                        value=sorted(supported_entities['Location']),
                        style={'marginBottom': '10px'}
                    )
                ]),
                style={'marginBottom': '10px'}
            )
        ]

        column3 = [
            dbc.Card(
                dbc.CardBody([
                    html.H5('Others', className="card-title"),
                    dcc.Checklist(
                        id='outros-selection',
                        options=[{'label': entity, 'value': entity} for entity in sorted(supported_entities['Others'])],
                        value=sorted(supported_entities['Others']),
                        style={'marginBottom': '10px'}
                    )
                ]),
                style={'marginBottom': '10px'}
            )
        ]

        # Organizar os cards em três colunas
        return html.Div([
            html.H3('Private Entity Recognition'),
            html.H4('Select Entities of Interest', style={'marginTop': '20px', 'marginBottom': '20px'}), 
            dbc.Row([
                dbc.Col(column1, width=4),
                dbc.Col(column2, width=4), 
                dbc.Col(column3, width=4),
            ]),
            html.Button('Process Entity Recognition', id='entity-process-button', n_clicks=0),
            dbc.Progress(id="entity-progress", striped=True, animated=True, style={"marginTop": "20px"}),
            html.Div(id="entity-time-estimate", style={'marginTop': '10px', 'fontSize': '14px'}),
            dcc.Loading(id="entity-loading", type="circle", children=[
                html.Div(id='entity-analysis-results'),
            ])
        ])
    
    # Adicione este código na aba de "TextStat Analysis" no método update_tab
    elif button_id == 'link-textstat':
        metrics_options = [
            {'label': 'Flesch Reading Ease', 'value': 'Flesch Reading Ease', 'description': 'This score measures the readability of the text. The higher the score, the easier it is to understand the text. A score between 90-100 is considered very easy to read, while a score between 0-29 is considered very confusing.'},
            {'label': 'SMOG Index', 'value': 'SMOG Index', 'description': 'The SMOG Index estimates the years of education a person needs to understand a piece of writing. It is particularly useful for assessing the readability of more complex texts.'},
            {'label': 'Flesch-Kincaid Grade Level', 'value': 'Flesch-Kincaid Grade Level', 'description': 'This metric indicates the U.S. school grade level required to understand the text. For example, a score of 9.3 suggests that a ninth grader would be able to read the document.'},
            {'label': 'Coleman-Liau Index', 'value': 'Coleman-Liau Index', 'description': 'Similar to other readability scores, the Coleman-Liau Index determines the grade level of the text but focuses on the number of letters per 100 words and the number of sentences per 100 words.'},
            {'label': 'Automated Readability Index', 'value': 'Automated Readability Index', 'description': 'ARI provides an estimate of the U.S. school grade level required to comprehend the text, based on characters per word and words per sentence.'},
            {'label': 'Dale-Chall Readability Score', 'value': 'Dale-Chall Readability Score', 'description': 'This score uses a list of 3,000 common words to assess readability, making it particularly effective for texts intended for younger audiences.'},
            {'label': 'Difficult Words', 'value': 'Difficult Words', 'description': 'This metric counts the number of words that are not common or easily understandable by readers below college level.'},
            {'label': 'Linsear Write Formula', 'value': 'Linsear Write Formula', 'description': 'This formula is used to calculate the U.S. school grade level of the text based on the number of easy and difficult words.'},
            {'label': 'Gunning Fog Index', 'value': 'Gunning Fog Index', 'description': 'The Gunning Fog Index estimates the years of formal education a reader needs to understand the text on the first reading. A score of 12 indicates high school level reading difficulty.'},
            {'label': 'Text Standard', 'value': 'Text Standard', 'description': 'Based on multiple readability tests, this metric provides an overall estimation of the grade level required to understand the text.'},
            {'label': 'Lexicon Count', 'value': 'Lexicon Count', 'description': 'This measures the number of words in the text, providing insight into the length and complexity of the content.'},
            {'label': 'Sentence Count', 'value': 'Sentence Count', 'description': 'This counts the number of sentences in the text, which can be indicative of the text’s complexity and structure.'},
            {'label': 'Syllable Count', 'value': 'Syllable Count', 'description': 'The number of syllables in the text can impact readability, with more syllables indicating more complex words.'},
            {'label': 'Character Count', 'value': 'Character Count', 'description': 'This measures the total number of characters in the text, offering another view of text length.'},
            {'label': 'Polysyllable Count', 'value': 'Polysyllable Count', 'description': 'This counts the number of words with three or more syllables, which typically indicates higher reading difficulty.'},
            {'label': 'Monosyllable Count', 'value': 'Monosyllable Count', 'description': 'This counts the number of one-syllable words, often associated with easier readability.'}
        ]
        
        # Divide as opções em três colunas
        col1_metrics = metrics_options[:6]
        col2_metrics = metrics_options[6:12]
        col3_metrics = metrics_options[12:]

        return html.Div([
            html.H3('TextStat Analysis'),
            html.Div([
                html.Label('Select Metrics to Display:', style={'marginBottom': '10px'}),
                dbc.Row([
                    dbc.Col([
                        dcc.Checklist(
                            id='textstat-metric-checklist-col1',
                            options=[{'label': [option['label'], html.Span('?', id=f"tooltip-{option['value'].replace(' ', '-')}", style={'color': 'black', 'backgroundColor': '#e0e0e0', 'borderRadius': '50%', 'padding': '2px 6px', 'marginLeft': '5px', 'cursor': 'pointer', 'display': 'inline-block', 'fontSize': '12px', 'textAlign': 'center'})], 'value': option['value']} for option in col1_metrics],
                            value=[option['value'] for option in col1_metrics],  # Seleciona todas as métricas da coluna 1
                            style={'marginBottom': '10px'},
                            labelStyle={'marginLeft': '10px', 'display': 'flex', 'alignItems': 'center'}  # Alinha checkbox e label
                        ),
                        # Tooltips para cada métrica
                        *[dbc.Tooltip(option['description'], target=f"tooltip-{option['value'].replace(' ', '-')}", placement='right') for option in col1_metrics]
                    ], width=4),
                    dbc.Col([
                        dcc.Checklist(
                            id='textstat-metric-checklist-col2',
                            options=[{'label': [option['label'], html.Span('?', id=f"tooltip-{option['value'].replace(' ', '-')}", style={'color': 'black', 'backgroundColor': '#e0e0e0', 'borderRadius': '50%', 'padding': '2px 6px', 'marginLeft': '5px', 'cursor': 'pointer', 'display': 'inline-block', 'fontSize': '12px', 'textAlign': 'center'})], 'value': option['value']} for option in col2_metrics],
                            value=[option['value'] for option in col2_metrics],  # Seleciona todas as métricas da coluna 2
                            style={'marginBottom': '10px'},
                            labelStyle={'marginLeft': '10px', 'display': 'flex', 'alignItems': 'center'}  # Alinha checkbox e label
                        ),
                        # Tooltips para cada métrica
                        *[dbc.Tooltip(option['description'], target=f"tooltip-{option['value'].replace(' ', '-')}", placement='right') for option in col2_metrics]
                    ], width=4),
                    dbc.Col([
                        dcc.Checklist(
                            id='textstat-metric-checklist-col3',
                            options=[{'label': [option['label'], html.Span('?', id=f"tooltip-{option['value'].replace(' ', '-')}", style={'color': 'black', 'backgroundColor': '#e0e0e0', 'borderRadius': '50%', 'padding': '2px 6px', 'marginLeft': '5px', 'cursor': 'pointer', 'display': 'inline-block', 'fontSize': '12px', 'textAlign': 'center'})], 'value': option['value']} for option in col3_metrics],
                            value=[option['value'] for option in col3_metrics],  # Seleciona todas as métricas da coluna 3
                            style={'marginBottom': '10px'},
                            labelStyle={'marginLeft': '10px', 'display': 'flex', 'alignItems': 'center'}  # Alinha checkbox e label
                        ),
                        # Tooltips para cada métrica
                        *[dbc.Tooltip(option['description'], target=f"tooltip-{option['value'].replace(' ', '-')}", placement='right') for option in col3_metrics]
                    ], width=4),
                ]),
            ]),
            html.Button('Process TextStat Analysis', id='textstat-process-button', n_clicks=0),
            dbc.Progress(id="textstat-progress", striped=True, animated=True, style={"marginTop": "20px"}),
            html.Div(id="textstat-time-estimate", style={'marginTop': '10px', 'fontSize': '14px'}),
            dcc.Loading(id="textstat-loading", type="circle", children=[
                html.Div(id='textstat-analysis-results'),
            ])
        ])





if __name__ == '__main__':
    app.run_server(debug=True)