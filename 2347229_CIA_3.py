import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import cv2
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load a sample of the dataset to reduce memory usage
@st.cache_data
def load_sample_data(sample_size=1000):
    data = pd.read_csv("WomensClothingE-CommerceReviews.csv")
    return data.sample(min(sample_size, len(data)))

data = load_sample_data()

# Define functions for text processing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]
    return ' '.join(tokens) 

def plot_3d(data, x_column, y_column, z_column):
    fig = plt.figure(figsize=(6, 6)) 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[x_column], data[y_column], data[z_column], c='purple', s=40)
    ax.set_xlabel(str(x_column))
    ax.set_ylabel(str(y_column))
    ax.set_zlabel(str(z_column))
    st.write(fig)

# Define function for image processing
# Define function for image processing
def process_image(image, technique):
    # Process the image based on the selected technique
    if technique == 'Original':
        return image
    elif technique == 'Resize':
        resized_image = cv2.resize(image, (200, 200))
        return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # Convert color format to RGB
    elif technique == 'Grayscale Conversion':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    elif technique == 'Image Cropping':
        roi = image[100:300, 100:300]  # Larger region for cropping
        return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert color format to RGB
    elif technique == 'Image Rotation':
        rotation_matrix = cv2.getRotationMatrix2D((100, 100), 45, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (200, 200))
        return cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)  # Convert color format to RGB
    elif technique == 'Applying Filters':
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # Increase the kernel size
        return cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)  # Convert color format to RGB


# Streamlit UI
st.title('Womens Clothing E-Commerce Reviews Analysis')
st.subheader("📊 Dataset")
st.write(data)

# Create tabs
tabs = ['3D Plot Visualization', 'Image Processing', 'Text Similarity Analysis']
choice = st.sidebar.selectbox('Select Analysis', tabs)

# Render different tabs based on user choice
if choice == '3D Plot Visualization':
    st.subheader('3D Plot Visualization')

    # Select columns for 3D plot
    x_column = st.selectbox('Select X-axis Column', data.columns)
    y_column = st.selectbox('Select Y-axis Column', data.columns)
    z_column = st.selectbox('Select Z-axis Column', data.columns)

    # Plot 3D graph
    plot_3d(data, x_column, y_column, z_column)
elif choice == 'Image Processing':
    st.title('Image Processing')

    # Image processing options
    image_options = ['Original', 'Resize', 'Grayscale Conversion', 'Image Cropping', 'Image Rotation', 'Applying Filters']
    selected_option = st.selectbox('Select Image Processing Technique', image_options)

    # Image selection
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_image is not None:
        # Read the uploaded image as an array
        img_array = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        original_image = cv2.imdecode(img_array, 1)

        # Display original image
        st.subheader('Original Image')
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Process image based on selected technique
        processed_image = process_image(original_image, selected_option)

        # Display processed image
        st.subheader(f"{selected_option} Image")
        st.image(processed_image, caption=f"{selected_option} Image", use_column_width=True)
elif choice == 'Text Similarity Analysis':
    st.subheader('Text Similarity Analysis')

    # Text analysis
    text_input = st.text_area("Enter your text:", "")
    processed_text = preprocess_text(text_input)

    # Display processed text
    if text_input:
        st.subheader("Processed Text:")
        st.write(processed_text)

        # Vectorize the processed text
        vectorizer = TfidfVectorizer()
        text_vectors = vectorizer.fit_transform([processed_text, *data['Review Text'].fillna('')])

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(text_vectors)[0][1:]

        # Plotting similarity scores
        fig, ax = plt.subplots()
        ax.barh(data.index, cosine_similarities)
        ax.set_yticks(data.index)
        ax.set_yticklabels(data['Review Text'].fillna('').apply(lambda x: x[:50] + '...'))
        ax.invert_yaxis()  # Invert y-axis to display the most similar at the top
        ax.set_xlabel('Cosine Similarity')
        ax.set_title('Text Similarity with User Input')
        st.pyplot(fig)

        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)
        st.subheader("Word Cloud")
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())  # Pass the current figure explicitly
