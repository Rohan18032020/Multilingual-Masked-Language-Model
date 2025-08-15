# This module implements an enhanced masked language model with support for multiple languages,
# fine-tuning capabilities, and various visualization and analysis tools.

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import AdamWeightDecay
from datasets import load_dataset, DatasetDict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# Dictionary mapping supported languages to their respective pre-trained BERT models
MODELS = {
    "english": "bert-base-uncased",
    "french": "camembert-base",
    "german": "dbmdz/bert-base-german-uncased",
    "chinese": "bert-base-chinese",
    "japanese": "cl-tohoku/bert-base-japanese"
}

# Number of top predictions to return
K = 3

# Visual constants for attention diagram generation
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


class EnhancedMaskedLanguageModel:
    """
    A class implementing enhanced masked language modeling with multi-language support,
    fine-tuning capabilities, and visualization tools.
    """
    def __init__(self, language):
        """
        Initialize the model for a specific language.
        
        Args:
            language (str): The language to use (must be one of the supported languages in MODELS)
        
        Raises:
            ValueError: If the specified language is not supported
        """
        if language not in MODELS:
            raise ValueError(f"Unsupported language. Please choose from {', '.join(MODELS.keys())}.")
        self.language = language
        self.model_name = MODELS[language]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = TFAutoModelForMaskedLM.from_pretrained(self.model_name)

    def predict(self, text):
        """
        Predict masked tokens in the input text.
        
        Args:
            text (str): Input text containing the mask token
            
        Returns:
            tuple: (predictions, attentions) where predictions is a list of (text, confidence) tuples
                  and attentions contains the attention weights
                  
        Raises:
            ValueError: If the input text doesn't contain a mask token
        """
        if self.tokenizer.mask_token not in text:
            raise ValueError(f"Input must include mask token {self.tokenizer.mask_token}.")

        inputs = self.tokenizer(text, return_tensors="tf")
        mask_token_index = self.get_mask_token_index(inputs)

        result = self.model(**inputs, output_attentions=True)
        mask_token_logits = result.logits[0, mask_token_index]
        top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
        
        predictions = []
        for token in top_tokens:
            predicted_text = text.replace(self.tokenizer.mask_token, self.tokenizer.decode([token]))
            confidence = tf.nn.softmax(mask_token_logits)[token].numpy()
            predictions.append((predicted_text, confidence))
        
        return predictions, result.attentions

    def get_mask_token_index(self, inputs):
        """
        Get the index of the mask token in the tokenized input.
        
        Args:
            inputs: Tokenized input containing the mask token
            
        Returns:
            int or None: Index of the mask token, or None if not found
        """
        mask_token_id = self.tokenizer.mask_token_id
        input_ids = inputs.input_ids[0].numpy()
        mask_positions = np.where(input_ids == mask_token_id)[0]
        
        if len(mask_positions) == 0:
            return None
        
        return mask_positions[0]
    
    def fine_tune(self, file_path, num_epochs=3, batch_size=8):
        """
        Fine-tune the model on custom data.
        
        Args:
            file_path (str): Path to the training data file
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Raises:
            FileNotFoundError: If the specified file_path doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Load and split the dataset
        datasets = load_dataset("text", data_files={"train": file_path})
        datasets = datasets["train"].train_test_split(test_size=0.2, seed=42)
        datasets = DatasetDict({
            "train": datasets["train"],
            "validation": datasets["test"]
        })

        # Create data collator for masked language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm_probability=0.15, 
            return_tensors="np"
        )

        # Benchmark initial model performance
        print("Benchmarking before fine-tuning...")
        initial_perplexity = self.benchmark(datasets["validation"]["text"])
        print(f"Initial perplexity: {initial_perplexity:.2f}")

        # Tokenization and text grouping functions
        def tokenize_function(examples):
            return self.tokenizer(examples["text"])

        block_size = 128

        def group_texts(examples):
            """Group texts into blocks of specified size."""
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Prepare datasets for training
        tokenized_datasets = datasets.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=1000,
        )

        # Prepare TensorFlow datasets
        train_set = self.model.prepare_tf_dataset(
            lm_datasets["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=data_collator,
        )

        validation_set = self.model.prepare_tf_dataset(
            lm_datasets["validation"],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=data_collator,
        )

        # Configure optimizer and compile model
        optimizer = AdamWeightDecay(lr=3e-5, weight_decay_rate=0.01)
        self.model.compile(optimizer=optimizer, jit_compile=True)

        print("Fitting Model...")
        # Perform fine-tuning
        self.model.fit(train_set, validation_data=validation_set, epochs=num_epochs)

        # Benchmark final model performance
        print("Benchmarking after fine-tuning...")
        final_perplexity = self.benchmark(datasets["validation"]["text"])
        print(f"Final perplexity: {final_perplexity:.2f}")

        improvement = (initial_perplexity - final_perplexity) / initial_perplexity * 100
        print(f"Perplexity improvement: {improvement:.2f}%")

    def benchmark(self, texts):
        """
        Calculate perplexity score for the model on given texts.
        
        Args:
            texts (list): List of text strings to evaluate
            
        Returns:
            float: Perplexity score
        """
        total_loss = 0
        total_tokens = 0

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True)
            labels = tf.identity(inputs["input_ids"])
            inputs["labels"] = labels

            outputs = self.model(**inputs)
            loss = outputs.loss
            total_loss += loss.numpy() * labels.shape[1]
            total_tokens += labels.shape[1]

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return np.mean(perplexity)

    def get_token_index(self, text):
        """
        Get the index of the mask token in the text.
        
        Args:
            text (str): Input text containing mask token
            
        Returns:
            int or None: Index of mask token, or None if not found
        """
        tokens = self.tokenizer.encode(text)
        try:
            return tokens.index(self.tokenizer.mask_token_id)
        except ValueError:
            return None

    def explain_prediction(self, text, K=5):
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            text (str): Input text containing mask token
            K (int): Number of top predictions to explain
            
        Returns:
            list: List of (token, importance) tuples
            
        Raises:
            ValueError: If text doesn't contain mask token
        """
        try:
            # Validation checks
            if self.tokenizer.mask_token not in text:
                raise ValueError(f"Input text must contain the mask token: {self.tokenizer.mask_token}")

            mask_token_index = self.get_token_index(text)
            if mask_token_index is None:
                raise ValueError("No mask token found in the input text.")
            
            # Get initial predictions
            initial_inputs = self.tokenizer(text, return_tensors="tf")
            initial_outputs = self.model(initial_inputs)
            initial_logits = initial_outputs.logits[0, mask_token_index, :]
            top_k_values, top_k_indices = tf.nn.top_k(initial_logits, k=K)
            top_k_indices = top_k_indices.numpy()
            
            # Define prediction function for SHAP
            def focused_predict(texts):
                outputs = []
                for t in texts:
                    inputs = self.tokenizer(t, return_tensors="tf", truncation=True, padding=True)
                    logits = self.model(inputs).logits
                    mask_positions = tf.where(inputs.input_ids[0] == self.tokenizer.mask_token_id)
                    if len(mask_positions) == 0:
                        curr_mask_idx = mask_token_index
                    else:
                        curr_mask_idx = int(mask_positions[0][0])
                    probs = tf.nn.softmax(logits[0, curr_mask_idx, :])
                    selected_probs = tf.gather(probs, top_k_indices)
                    outputs.append(selected_probs.numpy())
                return np.array(outputs)

            # Setup SHAP explainer
            token_names = [self.tokenizer.decode([idx]) for idx in top_k_indices]
            focused_predict.output_names = token_names
            masker = shap.maskers.Text(self.tokenizer)
            explainer = shap.Explainer(focused_predict, masker, output_names=token_names)

            # Generate SHAP values
            shap_values = explainer([text], fixed_context=1, batch_size=1)
            values = shap_values.values[0]
            if isinstance(values, list):
                values = np.array(values).flatten()

            # Generate visualizations
            # Bar Plot
            token_importances = np.abs(values).mean(axis=0)
            sorted_indices = np.argsort(token_importances)[::-1]
            sorted_token_names = np.array(token_names)[sorted_indices]
            sorted_importances = token_importances[sorted_indices]

            plt.figure(figsize=(10, 6))
            plt.barh(sorted_token_names, sorted_importances, color='skyblue')
            plt.gca().invert_yaxis()
            plt.title("Bar Plot of Token Importance")
            plt.xlabel("Mean Absolute SHAP Value (Importance)")
            plt.tight_layout()
            plt.savefig('shap_bar_plot.png', bbox_inches='tight', dpi=300)
            plt.close()

            # Waterfall Plot
            plt.figure(figsize=(10, 6))
            cleaned_data = [x if x != '' else '[PAD]' for x in shap_values.data[0]]
            single_prediction_values = shap_values.values[0, :, 0]
            exp = shap.Explanation(
                values=single_prediction_values,
                base_values=float(shap_values.base_values[0][0]),
                data=cleaned_data,
                feature_names=cleaned_data
            )
            shap.plots.waterfall(exp, show=False)
            plt.title("Waterfall Plot of SHAP Values")
            plt.tight_layout()
            plt.savefig('shap_waterfall.png', bbox_inches='tight', dpi=300)
            plt.close()

            # Decision Plot
            plt.figure(figsize=(10, 6))
            decision_values = values if values.ndim == 1 else values[:, 0]
            shap.decision_plot(
                base_value=float(shap_values.base_values[0][0]),
                shap_values=decision_values,
                feature_names=cleaned_data,
                show=False
            )
            plt.title("Decision Plot of SHAP Values")
            plt.tight_layout()
            plt.savefig('shap_decision_plot.png', bbox_inches='tight', dpi=300)
            plt.close()

            # Summary Plot
            plt.figure(figsize=(10, 5))
            shap.summary_plot(
                values,
                feature_names=token_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig('shap_summary.png', bbox_inches='tight', dpi=300)
            plt.close()

            print("SHAP visualizations saved successfully!")

            # Waterfall Plot
            token_importances = np.abs(values).mean(axis=0)
            results = list(zip(token_names, token_importances))
            results.sort(key=lambda x: x[1], reverse=True)

            return results

        except Exception as e:
            raise

    def make_prediction_function(self, mask_token_index):
        """
        Create a prediction function for SHAP explanations.
        
        Args:
            mask_token_index (int): Index of the mask token
            
        Returns:
            function: Prediction function compatible with SHAP
        """
        def predict(texts):
            outputs = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="tf", padding=True)
                logits = self.model(inputs).logits
                mask_logits = logits[0, mask_token_index, :]
                outputs.append(mask_logits.numpy())
            return np.array(outputs)
        predict.output_names = [self.tokenizer.decode([i]) for i in range(self.tokenizer.vocab_size)]
        return predict

    def get_color_for_attention_score(self, attention_score):
        """
        Return a tuple of three integers representing a shade of gray for the given `attention_score`.
        The attention score is scaled to be within [0, 255] where 0 is black and 255 is white.
        
        Args:
            attention_score (float): A normalized attention score between 0 and 1.
        
        Returns:
            tuple: A tuple of three integers representing an RGB color where all values are equal 
                   to give a gray shade based on the attention score.
        """
        num = int(attention_score * 255)
        return num, num, num

    def visualize_attentions(self, tokens, attentions):
        """
        Visualize the attention weights for each attention head in each layer by generating attention diagrams.
        
        Args:
            tokens (list): The tokenized input text as a list of token strings.
            attentions (tf.Tensor): A tensor containing attention weights from the model for each layer and head.
        
        This method loops through each attention head in each layer and generates attention diagrams.
        """
        for i in range(len(attentions)):
            for k in range(len(attentions[i][0])):
                self.generate_diagram(
                    i+1,
                    k+1,
                    tokens,
                    attentions[i][0][k]
                )

    def generate_diagram(self, layer_number, head_number, tokens, attention_weights):
        """
        Generate and save an attention diagram for a specific attention head and layer.
        
        Args:
            layer_number (int): The index of the layer.
            head_number (int): The index of the attention head within the layer.
            tokens (list): The tokenized input text as a list of token strings.
            attention_weights (np.ndarray): A matrix of attention weights between tokens.
        
        This method creates an image representation of the attention matrix, with token words on both axes.
        The intensity of each cell represents the attention weight between tokens.
        """
        #Set the image size based on the number of tokens
        image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
        img = Image.new("RGBA", (image_size, image_size), "black")
        draw = ImageDraw.Draw(img)

        # Draw each token onto the image
        for i, token in enumerate(tokens):
            # Draw token columns
            token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
            token_draw = ImageDraw.Draw(token_image)
            token_draw.text(
                (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
                token,
                fill="white",
                font=FONT
            )
            token_image = token_image.rotate(90)
            img.paste(token_image, mask=token_image)

            # Draw token rows
            _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
            draw.text(
                (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
                token,
                fill="white",
                font=FONT
            )

        # Draw each word
        for i in range(len(tokens)):
            y = PIXELS_PER_WORD + i * GRID_SIZE
            for j in range(len(tokens)):
                x = PIXELS_PER_WORD + j * GRID_SIZE
                color = self.get_color_for_attention_score(attention_weights[i][j])
                draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

        # Save image
        img.save(f"images/Attention_Layer{layer_number}_Head{head_number}.png")

    def analyze_contextual_embeddings(self, text):
        """
        Analyze and visualize the contextual embeddings of the input text using PCA or t-SNE.
        
        Args:
            text (str): The input text to analyze.
        
        This method tokenizes the input text, extracts the hidden state embeddings from the model,
        and reduces their dimensionality using either PCA (for a small number of tokens) or t-SNE 
        (for larger inputs). The resulting embeddings are visualized in a 2D scatter plot.
        """
        inputs = self.tokenizer(text, return_tensors="tf")
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1][0].numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Choose dimensionality reduction method based on number of tokens
        n_samples = last_hidden_states.shape[0]
        if n_samples < 4:
            print("Not enough tokens for meaningful visualization. Please use a longer text.")
            return
        
        # Use PCA for very small number of tokens
        elif n_samples < 50:
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(last_hidden_states)
            method = "PCA"

        # Use t-SNE for larger number of tokens
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples - 1))
            embeddings_2d = tsne.fit_transform(last_hidden_states)
            method = "t-SNE"

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(tokens)), cmap='viridis')
        
        # Add labels for a subset of points to avoid clutter
        max_labels = 30
        step = max(1, len(tokens) // max_labels)
        for i in range(0, len(tokens), step):
            plt.annotate(tokens[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
        
        plt.colorbar(scatter, label="Token Position")
        plt.title(f"Contextual Embeddings Visualization ({method})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig('contextual_embeddings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Contextual embeddings visualization saved as 'contextual_embeddings.png'")

def main():
    """
    Main function to interact with the EnhancedMaskedLanguageModel. It provides a command-line interface for the user
    to perform several actions such as predicting masked tokens, fine-tuning the model, analyzing contextual embeddings,
    and explaining predictions using SHAP values.

    The user is prompted to select a language model and then presented with a menu of options:
    1. Predict a masked token in a sentence.
    2. Fine-tune the language model on a custom dataset.
    3. Analyze contextual embeddings for a given input text.
    4. Explain a prediction using SHAP values.
    5. Exit the program.

    Each option corresponds to a specific function of the EnhancedMaskedLanguageModel class.

    The function handles any potential errors that occur during user input or model operations by catching exceptions 
    and safely terminating the program.
    """
    try:
        # Prompt user to select a language and initialize the model
        language = input("Select language (english/french/german/chinese/japanese): ").lower()
        model = EnhancedMaskedLanguageModel(language)

        while True:
            # Display menu options to the user
            print("\nOptions:")
            print("1. Predict masked token")
            print("2. Fine-tune model")
            print("3. Analyze contextual embeddings")
            print("4. Explain prediction")
            print("5. Exit")
            choice = input("Enter your choice (1-5): ")

            if choice == '1':
                # Option 1: Predict masked token
                text = input(f"Enter text in {language.capitalize()} with [MASK] token: ")
                predictions, attentions = model.predict(text)
                for pred, conf in predictions:
                    print(f"Prediction: {pred} (Confidence: {conf:.4f})")
                model.visualize_attentions(model.tokenizer.tokenize(text), attentions)

            elif choice == '2':
                # Option 2: Fine-tune the model with a user-specified dataset
                file_path = input("Enter the path to your fine-tuning dataset file: ")
                num_epochs = int(input("Enter number of epochs for fine-tuning: "))
                model.fine_tune(file_path, num_epochs)
                print("Fine-tuning completed.")

            elif choice == '3':
                # Option 3: Analyze contextual embeddings for the input text
                text = input(f"Enter text in {language.capitalize()} for embedding analysis: ")
                model.analyze_contextual_embeddings(text)

            elif choice == '4':
                # Option 4: Explain predictions using SHAP values
                text = input(f"Enter text in {language.capitalize()} with [MASK] token for explanation: ")
                explanations = model.explain_prediction(text)
                print("Top predictions and their SHAP values:")
                for word, value in explanations:
                    print(f"{word}: {value:.4f}")

            elif choice == '5':
                # Option 5: Exit the program
                print("Exiting the program.")
                break

            else:
                # Handle invalid input
                print("Invalid choice. Please try again.")

    except Exception as e:
        # Handle any exceptions and safely exit the program
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
