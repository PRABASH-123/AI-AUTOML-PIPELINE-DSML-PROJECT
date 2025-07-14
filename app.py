import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
import io
import os
from datetime import datetime
from io import BytesIO

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                            GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, r2_score, mean_squared_error, 
                            silhouette_score, classification_report, 
                            confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, 
                                   Flatten, Dropout, Embedding, LSTM, 
                                   Bidirectional, GlobalMaxPooling1D, Conv1D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import (VGG16, ResNet50, MobileNetV2)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import cv2

# Set page config with professional background
st.set_page_config(
    page_title="Advanced AutoML Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional background
def set_background():
    """Set a professional gradient background"""
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: rgba(255, 255, 255, 0.95);
        background-image: none;
    }
    .stButton>button {
        background-color: #4f8bf9;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3a6bb7;
    }
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.8);
    }
    </style>
    """, unsafe_allow_html=True)

set_background()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None
if 'image_data' not in st.session_state:
    st.session_state.image_data = {'images': None, 'labels': None}
if 'text_data' not in st.session_state:
    st.session_state.text_data = {'texts': None, 'labels': None}
if 'text_vectorizer' not in st.session_state:
    st.session_state.text_vectorizer = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

# Helper functions
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.session_state.data_type = 'structured'
            return df
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
            st.session_state.data_type = 'structured'
            return df
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            st.session_state.data_type = 'structured'
            return df
        elif uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(uploaded_file)
            st.session_state.data_type = 'image'
            # For demo, create dummy dataset with augmented versions
            img_array = np.array(img)
            augmented_images = [img_array]
            
            # Create augmented versions
            for angle in [90, 180, 270]:
                rows, cols = img_array.shape[0], img_array.shape[1]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                rotated = cv2.warpAffine(img_array, M, (cols, rows))
                augmented_images.append(rotated)
            
            st.session_state.image_data = {
                'images': augmented_images,
                'labels': [0] * len(augmented_images)  # Single class for demo
            }
            st.session_state.class_names = ['Class 0']
            return img
        elif uploaded_file.name.endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')
            st.session_state.data_type = 'text'
            # For demo, split text into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            st.session_state.text_data = {
                'texts': sentences[:10],  # Use first 10 sentences
                'labels': [0] * min(10, len(sentences))  # Single class for demo
            }
            st.session_state.class_names = ['Class 0']
            return text
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def describe_structured_data(df):
    st.subheader("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First 5 Rows**")
        st.write(df.head())
    
    with col2:
        st.write("**Data Types**")
        st.write(df.dtypes.astype(str))
    
    st.write("**Summary Statistics**")
    st.write(df.describe(include='all'))
    
    st.write("**Missing Values**")
    missing_data = df.isnull().sum().to_frame(name="Missing Values")
    missing_data["Percentage"] = (missing_data["Missing Values"] / len(df)) * 100
    st.write(missing_data)

def describe_image_data(img):
    st.subheader("Image Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.write("**Image Properties**")
        st.write(f"Format: {img.format}")
        st.write(f"Size: {img.size}")
        st.write(f"Mode: {img.mode}")
        
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            st.write("**Channel-wise Statistics**")
            for i, channel in enumerate(['Red', 'Green', 'Blue']):
                st.write(f"{channel} Channel - Min: {img_array[:,:,i].min()}, Max: {img_array[:,:,i].max()}, Mean: {img_array[:,:,i].mean():.2f}")
        
        st.write("**Pixel Intensity Distribution**")
        fig, ax = plt.subplots()
        if len(img_array.shape) == 3:
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
                ax.plot(hist, color=color)
            ax.legend(['Red', 'Green', 'Blue'])
        else:
            hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
            ax.plot(hist, color='k')
        ax.set_xlim([0, 256])
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

def describe_text_data(text):
    st.subheader("Text Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Text Sample**")
        st.text(text[:1000] + "..." if len(text) > 1000 else text)
    
    with col2:
        st.write("**Text Statistics**")
        st.write(f"Character count: {len(text)}")
        st.write(f"Word count: {len(text.split())}")
        st.write(f"Line count: {len(text.splitlines())}")
        
        words = text.split()
        word_freq = pd.Series(words).value_counts().head(20)
        st.write("**Top 20 Words**")
        fig, ax = plt.subplots()
        word_freq.plot(kind='bar', ax=ax)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

def plot_structured_data(df):
    st.subheader("Data Visualization")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    plot_type = st.selectbox("Select Plot Type", 
                           ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", 
                            "Correlation Heatmap", "Pair Plot", "Violin Plot"])
    
    if plot_type == "Histogram" and numeric_cols:
        selected_col = st.selectbox("Select Column for Histogram", numeric_cols)
        fig = px.histogram(df, x=selected_col, marginal="box", 
                          title=f"Distribution of {selected_col}",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Box Plot" and numeric_cols:
        selected_col = st.selectbox("Select Column for Box Plot", numeric_cols)
        fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}",
                    template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Scatter Plot" and len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis", numeric_cols)
        with col2:
            y_axis = st.selectbox("Select Y-axis", numeric_cols)
        
        color_option = st.selectbox("Color by", ["None"] + cat_cols)
        
        if color_option == "None":
            fig = px.scatter(df, x=x_axis, y=y_axis, 
                            title=f"{y_axis} vs {x_axis}",
                            template="plotly_white")
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option, 
                            title=f"{y_axis} vs {x_axis} colored by {color_option}",
                            template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Bar Chart" and cat_cols:
        selected_col = st.selectbox("Select Categorical Column", cat_cols)
        value_counts = df[selected_col].value_counts().reset_index()
        value_counts.columns = ['Category', 'Count']
        fig = px.bar(value_counts, x='Category', y='Count',
                    title=f"Distribution of {selected_col}",
                    template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Correlation Heatmap" and len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        fig = px.imshow(corr, 
                       text_auto=True, 
                       aspect="auto", 
                       title="Correlation Heatmap",
                       template="plotly_white",
                       color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Pair Plot" and len(numeric_cols) >= 2:
        sample_size = st.slider("Sample Size (for performance)", 100, 1000, 500)
        sample_df = df[numeric_cols].sample(sample_size)
        
        color_option = st.selectbox("Color by (Pair Plot)", ["None"] + cat_cols)
        
        if color_option == "None":
            fig = px.scatter_matrix(sample_df, 
                                  template="plotly_white",
                                  title="Pair Plot")
        else:
            fig = px.scatter_matrix(sample_df, 
                                  color=df[color_option],
                                  template="plotly_white",
                                  title=f"Pair Plot colored by {color_option}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Violin Plot" and numeric_cols:
        selected_col = st.selectbox("Select Numeric Column", numeric_cols)
        if cat_cols:
            category = st.selectbox("Select Category Column", cat_cols)
            fig = px.violin(df, x=category, y=selected_col, box=True,
                           title=f"Violin Plot of {selected_col} by {category}",
                           template="plotly_white")
        else:
            fig = px.violin(df, y=selected_col, box=True,
                          title=f"Violin Plot of {selected_col}",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

def preprocess_structured_data(df):
    st.subheader("Data Preprocessing")
    processed_df = df.copy()
    
    # Handle missing values
    st.write("### Missing Value Handling")
    missing_cols = processed_df.columns[processed_df.isnull().any()].tolist()
    
    if missing_cols:
        for col in missing_cols:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{col}** - {processed_df[col].isnull().sum()} missing values")
            with col2:
                strategy = st.selectbox(
                    f"Imputation strategy for {col}",
                    ["Drop rows", "Mean", "Median", "Mode", "Constant"],
                    key=f"missing_{col}"
                )
                
                if strategy == "Drop rows":
                    processed_df = processed_df.dropna(subset=[col])
                elif strategy == "Constant":
                    constant_value = st.text_input(f"Enter constant value for {col}", value="0", key=f"const_{col}")
                    try:
                        constant_value = pd.to_numeric(constant_value)
                    except:
                        pass
                    processed_df[col].fillna(constant_value, inplace=True)
                else:
                    if strategy == "Mean":
                        fill_value = processed_df[col].mean()
                    elif strategy == "Median":
                        fill_value = processed_df[col].median()
                    elif strategy == "Mode":
                        fill_value = processed_df[col].mode()[0]
                    
                    processed_df[col].fillna(fill_value, inplace=True)
    else:
        st.write("No missing values found in the dataset.")
    
    # Remove duplicates
    st.write("### Duplicate Handling")
    dup_count = processed_df.duplicated().sum()
    st.write(f"Found {dup_count} duplicate rows.")
    
    if dup_count > 0:
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        if remove_duplicates:
            processed_df = processed_df.drop_duplicates()
            st.write(f"Removed {dup_count} duplicates. New shape: {processed_df.shape}")
    
    # Feature encoding
    st.write("### Feature Encoding")
    cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in cat_cols:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{col}** - {len(processed_df[col].unique())} unique values")
        with col2:
            encoding = st.selectbox(
                f"Encoding for {col}",
                ["Label Encoding", "One-Hot Encoding", "Leave as is"],
                key=f"encode_{col}"
            )
            
            if encoding == "Label Encoding":
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col])
            elif encoding == "One-Hot Encoding":
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)
    
    # Feature scaling
    st.write("### Feature Scaling")
    numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if numeric_cols:
        scaling_method = st.selectbox(
            "Select scaling method for numeric features",
            ["Standard Scaler (mean=0, std=1)", "Min-Max Scaler (0-1 range)", "No Scaling"],
            index=2
        )
        
        if scaling_method != "No Scaling":
            scaler = StandardScaler() if scaling_method == "Standard Scaler (mean=0, std=1)" else MinMaxScaler()
            processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
    
    st.session_state.processed_df = processed_df
    st.write("### Processed Data Preview")
    st.write(processed_df.head())
    
    return processed_df

def preprocess_image_data():
    st.subheader("Image Preprocessing")
    
    st.write("### Image Augmentation Options")
    augmentation_options = {
        'rescale': 1./255,
        'rotation_range': st.slider("Rotation Range (degrees)", 0, 45, 10),
        'width_shift_range': st.slider("Width Shift Range", 0.0, 0.5, 0.1),
        'height_shift_range': st.slider("Height Shift Range", 0.0, 0.5, 0.1),
        'shear_range': st.slider("Shear Range", 0.0, 0.5, 0.1),
        'zoom_range': st.slider("Zoom Range", 0.0, 0.5, 0.1),
        'horizontal_flip': st.checkbox("Horizontal Flip", True),
        'vertical_flip': st.checkbox("Vertical Flip", False),
        'fill_mode': st.selectbox("Fill Mode", ["nearest", "constant", "reflect", "wrap"])
    }
    
    img_size = st.selectbox("Select Image Size", 
                          [(32, 32), (64, 64), (128, 128), (224, 224), (256, 256)],
                          index=2)
    
    batch_size = st.slider("Batch Size", 8, 64, 16)
    
    # Store settings
    st.session_state.image_preprocessing = {
        'augmentation': augmentation_options,
        'img_size': img_size,
        'batch_size': batch_size
    }
    
    st.success("Image preprocessing settings saved!")
    
    # Show sample augmented images
    if st.session_state.image_data['images'] is not None and len(st.session_state.image_data['images']) > 1:
        st.write("### Sample Augmented Images")
        cols = st.columns(4)
        for i, img in enumerate(st.session_state.image_data['images'][:4]):
            with cols[i % 4]:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)

def preprocess_text_data():
    st.subheader("Text Preprocessing")
    
    st.write("### Text Vectorization")
    vectorization_method = st.selectbox(
        "Select vectorization method",
        ["TF-IDF", "Count Vectorizer", "Word Embeddings"]
    )
    
    if vectorization_method in ["TF-IDF", "Count Vectorizer"]:
        max_features = st.slider("Max Features", 100, 10000, 1000)
        ngram_range = (st.slider("Min N-gram", 1, 3, 1),
                      st.slider("Max N-gram", 1, 3, 1))
        
        if vectorization_method == "TF-IDF":
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        else:
            vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        # Fit vectorizer
        X = vectorizer.fit_transform(st.session_state.text_data['texts'])
        st.session_state.text_vectorizer = vectorizer
        st.session_state.text_features = X
        
        st.write(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        st.write("Sample transformed features:")
        st.write(pd.DataFrame(X[:5].toarray(), columns=vectorizer.get_feature_names_out()).head())
    
    else:  # Word Embeddings
        max_words = st.slider("Max Words", 1000, 50000, 5000)
        max_len = st.slider("Max Sequence Length", 50, 500, 100)
        embedding_dim = st.slider("Embedding Dimension", 50, 300, 100)
        
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(st.session_state.text_data['texts'])
        sequences = tokenizer.texts_to_sequences(st.session_state.text_data['texts'])
        X = pad_sequences(sequences, maxlen=max_len)
        
        st.session_state.tokenizer = tokenizer
        st.session_state.text_features = X
        st.session_state.text_embedding_dim = embedding_dim
        
        st.write(f"Vocabulary size: {len(tokenizer.word_index)}")
        st.write("Sample transformed sequences:")
        st.write(X[:5])
    
    st.session_state.text_preprocessing = {
        'method': vectorization_method,
        'max_features': max_features if vectorization_method != "Word Embeddings" else max_words,
        'max_len': max_len if vectorization_method == "Word Embeddings" else None,
        'embedding_dim': embedding_dim if vectorization_method == "Word Embeddings" else None
    }
    
    st.success("Text preprocessing completed!")

def split_structured_data(df):
    st.subheader("Train-Test Split")
    
    target_col = st.selectbox("Select Target Variable", df.columns)
    
    test_size = st.slider("Test Size Percentage", 10, 40, 20) / 100
    random_state = st.number_input("Random State", value=42)
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # For classification, get class names
    if str(y.dtype) in ['object', 'category']:
        st.session_state.class_names = y.unique().tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    st.write(f"Training set shape: {X_train.shape}")
    st.write(f"Testing set shape: {X_test.shape}")
    
    # Store in session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.target_col = target_col
    
    return X_train, X_test, y_train, y_test

def get_model_options(data_type, problem_type=None):
    model_options = {}
    
    if data_type == 'structured':
        if problem_type == "Classification":
            model_options["Supervised"] = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "SVM Classifier": SVC(probability=True),
                "Gradient Boosting Classifier": GradientBoostingClassifier()
            }
            model_options["Neural Networks"] = {
                "MLP Classifier": "mlp"
            }
        elif problem_type == "Regression":
            model_options["Supervised"] = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "SVM Regressor": SVR(),
                "Gradient Boosting Regressor": GradientBoostingRegressor()
            }
            model_options["Neural Networks"] = {
                "MLP Regressor": "mlp"
            }
        model_options["Unsupervised"] = {
            "K-Means Clustering": KMeans(),
            "DBSCAN": DBSCAN(),
            "PCA": "pca"
        }
    
    elif data_type == 'image':
        model_options["Deep Learning"] = {
            "CNN (Simple)": "simple_cnn",
            "VGG16 (Transfer Learning)": "vgg16",
            "ResNet50 (Transfer Learning)": "resnet50",
            "MobileNetV2 (Transfer Learning)": "mobilenet"
        }
    
    elif data_type == 'text':
        model_options["Traditional ML"] = {
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(probability=True),
            "Logistic Regression": LogisticRegression()
        }
        model_options["Deep Learning"] = {
            "LSTM": "lstm",
            "BiLSTM": "bilstm",
            "CNN for Text": "text_cnn"
        }
    
    return model_options

def build_mlp(input_shape, output_units, problem_type, params):
    model = Sequential()
    
    # Add hidden layers
    for i in range(params['hidden_layers']):
        if i == 0:
            model.add(Dense(params['units'], activation=params['activation'], input_shape=input_shape))
        else:
            model.add(Dense(params['units'], activation=params['activation']))
        model.add(Dropout(params['dropout']))
    
    # Add output layer
    if problem_type == "Classification":
        activation = 'softmax' if output_units > 1 else 'sigmoid'
        loss = 'categorical_crossentropy' if output_units > 1 else 'binary_crossentropy'
    else:
        activation = 'linear'
        loss = 'mse'
    
    model.add(Dense(output_units, activation=activation))
    
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model

def build_cnn(input_shape, num_classes, params):
    model = Sequential()
    
    model.add(Conv2D(params['filters'], (params['kernel_size'], params['kernel_size']), 
                    activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((params['pool_size'], params['pool_size'])))
    model.add(Conv2D(params['filters']*2, (params['kernel_size'], params['kernel_size']), activation='relu'))
    model.add(MaxPooling2D((params['pool_size'], params['pool_size'])))
    model.add(Flatten())
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_transfer_model(base_model_name, input_shape, num_classes, params):
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze layers
    for layer in base_model.layers[:-params['trainable_layers']]:
        layer.trainable = False
    
    # Add custom head
    x = base_model.output
    x = Flatten()(x)
    x = Dense(params['dense_units'], activation='relu')(x)
    x = Dropout(params['dropout'])(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_text_model(model_type, input_shape, num_classes, params):
    model = Sequential()
    
    if model_type == 'lstm':
        model.add(Embedding(input_shape[0], params['embedding_dim'], input_length=input_shape[1]))
        model.add(LSTM(params['lstm_units']))
        model.add(Dropout(params['dropout']))
        model.add(Dense(params['dense_units'], activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif model_type == 'bilstm':
        model.add(Embedding(input_shape[0], params['embedding_dim'], input_length=input_shape[1]))
        model.add(Bidirectional(LSTM(params['lstm_units'])))
        model.add(Dropout(params['dropout']))
        model.add(Dense(params['dense_units'], activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
    elif model_type == 'text_cnn':
        model.add(Embedding(input_shape[0], params['embedding_dim'], input_length=input_shape[1]))
        model.add(Conv1D(params['filters'], params['kernel_size'], activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(params['dense_units'], activation='relu'))
        model.add(Dropout(params['dropout']))
        model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_structured_model(X_train, y_train):
    st.subheader("Model Selection and Training")
    
    # Determine problem type
    problem_type = "Classification" if str(y_train.dtype) in ['object', 'category'] else "Regression"
    st.write(f"Detected Problem Type: **{problem_type}**")
    
    model_options = get_model_options('structured', problem_type)
    algorithm_type = st.selectbox("Select Algorithm Type", list(model_options.keys()))
    selected_model = st.selectbox("Select Model", list(model_options[algorithm_type].keys()))
    
    model = model_options[algorithm_type][selected_model]
    
    # Model hyperparameters
    st.write("### Model Hyperparameters")
    
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        max_depth = st.slider("Max Depth", 1, 20, 5)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        model.set_params(max_depth=max_depth, min_samples_split=min_samples_split)
    
    elif isinstance(model, (RandomForestClassifier, RandomForestRegressor, 
                     GradientBoostingClassifier, GradientBoostingRegressor)):
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        max_depth = st.slider("Max Depth", 1, 20, 5)
        model.set_params(n_estimators=n_estimators, max_depth=max_depth)
    
    elif isinstance(model, (SVC, SVR)):
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        C = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
        model.set_params(kernel=kernel, C=C)
    
    elif isinstance(model, KMeans):
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        model.set_params(n_clusters=n_clusters)
    
    elif selected_model in ["MLP Classifier", "MLP Regressor"]:
        st.session_state.model_type = "mlp"
        st.session_state.mlp_params = {
            'hidden_layers': st.slider("Number of Hidden Layers", 1, 5, 2),
            'units': st.slider("Units per Layer", 16, 256, 64),
            'activation': st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"]),
            'dropout': st.slider("Dropout Rate", 0.0, 0.5, 0.2),
            'learning_rate': st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        }
        
        if st.button("Train Model"):
            with st.spinner("Building and training neural network..."):
                try:
                    input_shape = (X_train.shape[1],)
                    output_units = len(st.session_state.class_names) if problem_type == "Classification" else 1
                    
                    model = build_mlp(input_shape, output_units, problem_type, st.session_state.mlp_params)
                    
                    # Convert y for classification
                    if problem_type == "Classification":
                        y_train_encoded = to_categorical(y_train)
                        y_test_encoded = to_categorical(st.session_state.y_test)
                    else:
                        y_train_encoded = y_train
                        y_test_encoded = st.session_state.y_test
                    
                    epochs = st.slider("Epochs", 10, 100, 20)
                    batch_size = st.slider("Batch Size", 16, 128, 32)
                    
                    history = model.fit(
                        X_train, y_train_encoded,
                        validation_data=(st.session_state.X_test, y_test_encoded),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0
                    )
                    
                    st.session_state.model = model
                    st.session_state.history = history.history
                    st.success("Neural network trained successfully!")
                    
                    # Plot training history
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name='Train Loss'
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name='Validation Loss'
                    ))
                    fig.update_layout(
                        title='Training History',
                        xaxis_title='Epochs',
                        yaxis_title='Loss',
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    return model
                except Exception as e:
                    st.error(f"Error training neural network: {e}")
                    return None
        return None
    
    elif selected_model == "PCA":
        st.session_state.model_type = "pca"
        n_components = st.slider("Number of Components", 2, min(50, X_train.shape[1]), 2)
        st.session_state.pca_params = {'n_components': n_components}
        
        if st.button("Train Model"):
            with st.spinner("Fitting PCA..."):
                try:
                    pca = PCA(n_components=n_components)
                    pca.fit(X_train)
                    
                    st.session_state.model = pca
                    st.success("PCA fitted successfully!")
                    
                    # Plot explained variance
                    fig = px.bar(
                        x=[f"PC{i+1}" for i in range(n_components)],
                        y=pca.explained_variance_ratio_,
                        title="Explained Variance Ratio",
                        labels={'x': 'Principal Component', 'y': 'Variance Ratio'},
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    return pca
                except Exception as e:
                    st.error(f"Error fitting PCA: {e}")
                    return None
        return None
    
    # Train the model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                if algorithm_type != "Unsupervised":
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train)
                
                st.session_state.model = model
                st.session_state.model_type = "traditional"
                st.success("Model trained successfully!")
                
                # Show model parameters
                st.write("### Model Parameters")
                st.write(model.get_params())
                
                return model
            except Exception as e:
                st.error(f"Error training model: {e}")
                return None
    return None

def train_image_model():
    st.subheader("Image Model Selection and Training")
    
    model_options = get_model_options('image')
    selected_model = st.selectbox("Select Model", list(model_options["Deep Learning"].keys()))
    
    # Model hyperparameters
    st.write("### Model Hyperparameters")
    
    if selected_model == "CNN (Simple)":
        st.session_state.model_type = "simple_cnn"
        st.session_state.cnn_params = {
            'filters': st.slider("Number of Filters", 16, 128, 32),
            'kernel_size': st.slider("Kernel Size", 3, 7, 3),
            'pool_size': st.slider("Pool Size", 2, 4, 2),
            'dense_units': st.slider("Dense Units", 32, 256, 128),
            'dropout': st.slider("Dropout Rate", 0.0, 0.5, 0.2),
            'learning_rate': st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        }
    
    elif selected_model in ["VGG16 (Transfer Learning)", "ResNet50 (Transfer Learning)", 
                          "MobileNetV2 (Transfer Learning)"]:
        base_model_name = selected_model.split()[0].lower()
        st.session_state.model_type = base_model_name
        st.session_state.transfer_params = {
            'trainable_layers': st.slider("Trainable Layers", 0, 20, 5),
            'dense_units': st.slider("Dense Units", 32, 256, 128),
            'dropout': st.slider("Dropout Rate", 0.0, 0.5, 0.2),
            'learning_rate': st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        }
    
    epochs = st.slider("Epochs", 5, 50, 10)
    batch_size = st.slider("Batch Size", 8, 64, 16)
    
    if st.button("Train Model"):
        with st.spinner("Training image model..."):
            try:
                # Prepare data
                images = st.session_state.image_data['images']
                labels = st.session_state.image_data['labels']
                
                # Convert to numpy arrays
                X = np.array(images)
                y = to_categorical(labels)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Resize images
                img_size = st.session_state.image_preprocessing['img_size']
                X_train_resized = np.array([cv2.resize(img, img_size) for img in X_train])
                X_test_resized = np.array([cv2.resize(img, img_size) for img in X_test])
                
                # Normalize
                X_train_resized = X_train_resized / 255.0
                X_test_resized = X_test_resized / 255.0
                
                # Build model
                input_shape = (img_size[0], img_size[1], 3)
                num_classes = len(st.session_state.class_names)
                
                if st.session_state.model_type == "simple_cnn":
                    model = build_cnn(input_shape, num_classes, st.session_state.cnn_params)
                else:
                    model = build_transfer_model(
                        st.session_state.model_type,
                        input_shape,
                        num_classes,
                        st.session_state.transfer_params
                    )
                
                # Train model
                history = model.fit(
                    X_train_resized, y_train,
                    validation_data=(X_test_resized, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
                
                st.session_state.model = model
                st.session_state.history = history.history
                st.session_state.X_test_img = X_test_resized
                st.session_state.y_test_img = y_test
                
                st.success("Image model trained successfully!")
                
                # Plot training history
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history.history['accuracy'],
                    name='Train Accuracy'
                ))
                fig.add_trace(go.Scatter(
                    y=history.history['val_accuracy'],
                    name='Validation Accuracy'
                ))
                fig.update_layout(
                    title='Training History',
                    xaxis_title='Epochs',
                    yaxis_title='Accuracy',
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                return True
            except Exception as e:
                st.error(f"Error training image model: {e}")
                return False
    return False

def train_text_model():
    st.subheader("Text Model Selection and Training")
    
    model_options = get_model_options('text')
    algorithm_type = st.selectbox("Select Algorithm Type", list(model_options.keys()))
    selected_model = st.selectbox("Select Model", list(model_options[algorithm_type].keys()))
    
    # Model hyperparameters
    st.write("### Model Hyperparameters")
    
    if algorithm_type == "Traditional ML":
        model = model_options[algorithm_type][selected_model]
        
        if isinstance(model, (SVC, LogisticRegression)):
            C = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
            model.set_params(C=C)
        
        if st.button("Train Model"):
            with st.spinner("Training text model..."):
                try:
                    X = st.session_state.text_features
                    y = st.session_state.text_data['labels']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    model.fit(X_train, y_train)
                    
                    st.session_state.model = model
                    st.session_state.X_test_text = X_test
                    st.session_state.y_test_text = y_test
                    st.session_state.model_type = "traditional_text"
                    
                    st.success("Text model trained successfully!")
                    return True
                except Exception as e:
                    st.error(f"Error training text model: {e}")
                    return False
    
    else:  # Deep Learning
        if selected_model in ["LSTM", "BiLSTM", "CNN for Text"]:
            model_type = selected_model.lower()
            st.session_state.model_type = model_type
            st.session_state.text_dl_params = {
                'lstm_units': st.slider("LSTM Units", 32, 256, 64),
                'filters': st.slider("Number of Filters", 32, 256, 64),
                'kernel_size': st.slider("Kernel Size", 3, 7, 5),
                'dense_units': st.slider("Dense Units", 32, 256, 64),
                'dropout': st.slider("Dropout Rate", 0.0, 0.5, 0.2),
                'learning_rate': st.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f"),
                'embedding_dim': st.session_state.text_embedding_dim
            }
        
        epochs = st.slider("Epochs", 5, 50, 10)
        batch_size = st.slider("Batch Size", 16, 128, 32)
        
        if st.button("Train Model"):
            with st.spinner("Training text model..."):
                try:
                    X = st.session_state.text_features
                    y = to_categorical(st.session_state.text_data['labels'])
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Build model
                    input_shape = X_train.shape[1:]
                    num_classes = len(st.session_state.class_names)
                    
                    model = build_text_model(
                        st.session_state.model_type,
                        input_shape,
                        num_classes,
                        st.session_state.text_dl_params
                    )
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1
                    )
                    
                    st.session_state.model = model
                    st.session_state.history = history.history
                    st.session_state.X_test_text = X_test
                    st.session_state.y_test_text = y_test
                    
                    st.success("Text model trained successfully!")
                    
                    # Plot training history
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['accuracy'],
                        name='Train Accuracy'
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_accuracy'],
                        name='Validation Accuracy'
                    ))
                    fig.update_layout(
                        title='Training History',
                        xaxis_title='Epochs',
                        yaxis_title='Accuracy',
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    return True
                except Exception as e:
                    st.error(f"Error training text model: {e}")
                    return False
    return False

def evaluate_structured_model(model, X_test, y_test):
    st.subheader("Model Evaluation")
    
    if model is None:
        st.warning("No trained model available. Please train a model first.")
        return
    
    # Determine problem type
    problem_type = "Classification" if str(y_test.dtype) in ['object', 'category'] else "Regression"
    
    # Make predictions
    try:
        if isinstance(model, (KMeans, DBSCAN)):
            st.warning("Unsupervised learning evaluation metrics are limited.")
            predictions = model.predict(X_test)
            
            if isinstance(model, KMeans):
                score = silhouette_score(X_test, predictions)
                st.write(f"Silhouette Score: {score:.4f}")
            
            # Plot clusters (for 2D data)
            if X_test.shape[1] >= 2:
                fig = px.scatter(
                    x=X_test.iloc[:, 0], 
                    y=X_test.iloc[:, 1], 
                    color=predictions,
                    title="Cluster Visualization",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif isinstance(model, PCA):
            st.write("### PCA Results")
            components = model.transform(X_test)
            st.write(f"Explained Variance Ratio: {model.explained_variance_ratio_}")
            
            if model.n_components_ >= 2:
                fig = px.scatter(
                    x=components[:, 0], 
                    y=components[:, 1], 
                    title="PCA Components Visualization",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.model_type == "mlp":
            # Neural network evaluation
            if problem_type == "Classification":
                y_test_encoded = to_categorical(y_test)
                loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
                st.metric("Test Accuracy", f"{accuracy:.4f}")
                st.metric("Test Loss", f"{loss:.4f}")
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_test_encoded, axis=1)
                
                # Confusion matrix
                cm = confusion_matrix(y_true_classes, y_pred_classes)
                fig = px.imshow(cm, text_auto=True, 
                               labels=dict(x="Predicted", y="Actual"),
                               title="Confusion Matrix",
                               template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification report
                st.write("### Classification Report")
                report = classification_report(y_true_classes, y_pred_classes, 
                                             target_names=st.session_state.class_names,
                                             output_dict=True)
                st.write(pd.DataFrame(report).transpose())
            else:
                y_pred = model.predict(X_test).flatten()
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                st.metric("R-squared", f"{r2:.4f}")
                st.metric("Mean Squared Error", f"{mse:.4f}")
                st.metric("Root Mean Squared Error", f"{rmse:.4f}")
                
                # Actual vs Predicted plot
                st.write("### Actual vs Predicted Values")
                fig = px.scatter(
                    x=y_test, 
                    y=y_pred, 
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Actual vs Predicted Values",
                    template="plotly_white"
                )
                fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                            x1=y_test.max(), y1=y_test.max(), 
                            line=dict(color="Red", width=2, dash="dot"))
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Traditional ML models
            predictions = model.predict(X_test)
            
            if problem_type == "Classification":
                st.write("### Classification Metrics")
                
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average='weighted')
                recall = recall_score(y_test, predictions, average='weighted')
                f1 = f1_score(y_test, predictions, average='weighted')
                
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
                
                # Confusion matrix
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, predictions)
                fig = px.imshow(cm, text_auto=True, 
                               labels=dict(x="Predicted", y="Actual"),
                               title="Confusion Matrix",
                               template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification report
                st.write("### Classification Report")
                report = classification_report(y_test, predictions, 
                                             target_names=st.session_state.class_names,
                                             output_dict=True)
                st.write(pd.DataFrame(report).transpose())
            
            else:
                st.write("### Regression Metrics")
                
                r2 = r2_score(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                
                st.metric("R-squared", f"{r2:.4f}")
                st.metric("Mean Squared Error", f"{mse:.4f}")
                st.metric("Root Mean Squared Error", f"{rmse:.4f}")
                
                # Actual vs Predicted plot
                st.write("### Actual vs Predicted Values")
                fig = px.scatter(
                    x=y_test, 
                    y=predictions, 
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Actual vs Predicted Values",
                    template="plotly_white"
                )
                fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                            x1=y_test.max(), y1=y_test.max(), 
                            line=dict(color="Red", width=2, dash="dot"))
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

def evaluate_image_model():
    st.subheader("Image Model Evaluation")
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("No trained model available. Please train a model first.")
        return
    
    model = st.session_state.model
    X_test = st.session_state.X_test_img
    y_test = st.session_state.y_test_img
    
    try:
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.metric("Test Accuracy", f"{accuracy:.4f}")
        st.metric("Test Loss", f"{loss:.4f}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Confusion matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        fig = px.imshow(cm, text_auto=True, 
                       labels=dict(x="Predicted", y="Actual"),
                       title="Confusion Matrix",
                       template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.write("### Classification Report")
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=st.session_state.class_names,
                                     output_dict=True)
        st.write(pd.DataFrame(report).transpose())
        
        # Sample predictions
        st.write("### Sample Predictions")
        sample_indices = np.random.choice(len(X_test), min(4, len(X_test)), replace=False)
        cols = st.columns(4)
        
        for i, idx in enumerate(sample_indices):
            with cols[i % 4]:
                st.image(X_test[idx], caption=f"True: {st.session_state.class_names[y_true_classes[idx]]}\nPred: {st.session_state.class_names[y_pred_classes[idx]]}",
                        use_column_width=True)
                st.write(f"Confidence: {np.max(y_pred[idx]):.2f}")
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

def evaluate_text_model():
    st.subheader("Text Model Evaluation")
    
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("No trained model available. Please train a model first.")
        return
    
    model = st.session_state.model
    X_test = st.session_state.X_test_text
    y_test = st.session_state.y_test_text
    
    try:
        if st.session_state.model_type == "traditional_text":
            # Traditional ML model evaluation
            predictions = model.predict(X_test)
            
            st.write("### Classification Metrics")
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")
            st.metric("Recall", f"{recall:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
            
            # Confusion matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)
            fig = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual"),
                           title="Confusion Matrix",
                           template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            st.write("### Classification Report")
            report = classification_report(y_test, predictions, 
                                         target_names=st.session_state.class_names,
                                         output_dict=True)
            st.write(pd.DataFrame(report).transpose())
            
            # Sample predictions
            st.write("### Sample Predictions")
            sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
            for idx in sample_indices:
                text = st.session_state.text_data['texts'][idx]
                st.write(f"**Text:** {text[:200]}...")
                st.write(f"**True Label:** {st.session_state.class_names[y_test[idx]]}")
                st.write(f"**Predicted Label:** {st.session_state.class_names[predictions[idx]]}")
                st.write("---")
        
        else:
            # Deep learning model evaluation
            y_test_cat = to_categorical(y_test)
            loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
            st.metric("Test Accuracy", f"{accuracy:.4f}")
            st.metric("Test Loss", f"{loss:.4f}")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test_cat, axis=1)
            
            # Confusion matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            fig = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual"),
                           title="Confusion Matrix",
                           template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            st.write("### Classification Report")
            report = classification_report(y_true_classes, y_pred_classes, 
                                         target_names=st.session_state.class_names,
                                         output_dict=True)
            st.write(pd.DataFrame(report).transpose())
            
            # Sample predictions
            st.write("### Sample Predictions")
            sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
            for idx in sample_indices:
                if st.session_state.text_preprocessing['method'] == "Word Embeddings":
                    # Reconstruct text from sequence
                    tokenizer = st.session_state.tokenizer
                    sequence = X_test[idx]
                    text = " ".join([tokenizer.index_word.get(i, "?") for i in sequence if i != 0])
                else:
                    # Get original text
                    text = st.session_state.text_data['texts'][idx]
                
                st.write(f"**Text:** {text[:200]}...")
                st.write(f"**True Label:** {st.session_state.class_names[y_true_classes[idx]]}")
                st.write(f"**Predicted Label:** {st.session_state.class_names[y_pred_classes[idx]]}")
                st.write(f"**Confidence:** {np.max(y_pred[idx]):.2f}")
                st.write("---")
    except Exception as e:
        st.error(f"Error during evaluation: {e}")

def save_model():
    st.subheader("Save Model")
    
    if 'model' not in st.session_state:
        st.warning("No trained model to save.")
        return
    
    model_name = st.text_input("Enter model name", "my_model")
    save_format = st.selectbox("Select save format", ["HDF5 (.h5)", "SavedModel", "Pickle"])
    
    if st.button("Save Model"):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}"
            
            if save_format == "HDF5 (.h5)":
                filename += ".h5"
                if isinstance(st.session_state.model, tf.keras.Model):
                    st.session_state.model.save(filename)
                else:
                    st.error("HDF5 format only supports Keras models")
                    return
            elif save_format == "SavedModel":
                filename += ""
                if isinstance(st.session_state.model, tf.keras.Model):
                    st.session_state.model.save(filename)
                else:
                    st.error("SavedModel format only supports Keras models")
                    return
            else:  # Pickle
                filename += ".pkl"
                if isinstance(st.session_state.model, tf.keras.Model):
                    st.error("Pickle format doesn't support Keras models. Use HDF5 or SavedModel instead.")
                    return
                else:
                    joblib.dump(st.session_state.model, filename)
            
            # Create download link
            with open(filename, "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            st.success(f"Model saved successfully as {filename}")
        except Exception as e:
            st.error(f"Error saving model: {e}")

# Main app
def main():
    st.title("🚀 Advanced AutoML Pipeline")
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">End-to-end automated machine learning for structured and unstructured data</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        st.title("Navigation")
        app_page = st.radio("", 
                           ["📤 Data Upload", 
                            "🔍 Data Exploration", 
                            "⚙️ Data Preprocessing", 
                            "✂️ Train-Test Split", 
                            "🤖 Model Training", 
                            "📊 Model Evaluation", 
                            "💾 Save Model"])
    
    # Page routing
    if app_page == "📤 Data Upload":
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset", 
                                       type=["csv", "json", "xlsx", "xls", "png", "jpg", "jpeg", "txt"])
        
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            if data is not None:
                if st.session_state.data_type == 'structured':
                    st.session_state.df = data
                    st.success("Structured data loaded successfully!")
                elif st.session_state.data_type == 'image':
                    st.success("Image loaded successfully!")
                elif st.session_state.data_type == 'text':
                    st.success("Text data loaded successfully!")
    
    elif app_page == "🔍 Data Exploration":
        st.header("Data Exploration")
        
        if st.session_state.data_type == 'structured' and st.session_state.df is not None:
            describe_structured_data(st.session_state.df)
            plot_structured_data(st.session_state.df)
        elif st.session_state.data_type == 'image' and st.session_state.image_data is not None:
            describe_image_data(Image.fromarray(st.session_state.image_data['images'][0]))
        elif st.session_state.data_type == 'text' and st.session_state.text_data is not None:
            describe_text_data(st.session_state.text_data['texts'][0])
        else:
            st.warning("Please upload data first.")
    
    elif app_page == "⚙️ Data Preprocessing":
        st.header("Data Preprocessing")
        
        if st.session_state.data_type == 'structured' and st.session_state.df is not None:
            preprocess_structured_data(st.session_state.df)
        elif st.session_state.data_type == 'image' and st.session_state.image_data is not None:
            preprocess_image_data()
        elif st.session_state.data_type == 'text' and st.session_state.text_data is not None:
            preprocess_text_data()
        else:
            st.warning("Please upload data first.")
    
    elif app_page == "✂️ Train-Test Split":
        st.header("Train-Test Split")
        
        if st.session_state.data_type == 'structured' and st.session_state.processed_df is not None:
            split_structured_data(st.session_state.processed_df)
        elif st.session_state.data_type in ['image', 'text']:
            st.warning("For image/text data, the split is handled during model training")
        else:
            st.warning("Please preprocess data first.")
    
    elif app_page == "🤖 Model Training":
        st.header("Model Training")
        
        if st.session_state.data_type == 'structured' and st.session_state.X_train is not None:
            train_structured_model(st.session_state.X_train, st.session_state.y_train)
        elif st.session_state.data_type == 'image' and st.session_state.image_data is not None:
            train_image_model()
        elif st.session_state.data_type == 'text' and st.session_state.text_data is not None:
            train_text_model()
        else:
            st.warning("Please prepare data first.")
    
    elif app_page == "📊 Model Evaluation":
        st.header("Model Evaluation")
        
        if st.session_state.data_type == 'structured' and st.session_state.model is not None:
            evaluate_structured_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test)
        elif st.session_state.data_type == 'image' and 'model' in st.session_state:
            evaluate_image_model()
        elif st.session_state.data_type == 'text' and 'model' in st.session_state:
            evaluate_text_model()
        else:
            st.warning("Please train a model first.")
    
    elif app_page == "💾 Save Model":
        save_model()

if __name__ == "__main__":
    main()