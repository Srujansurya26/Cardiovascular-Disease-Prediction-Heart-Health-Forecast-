import numpy as np
import streamlit as st
from Ecg import ECG
svm_probs = None
knn_probs = None
rf_probs = None
bayes_probs = None
logistic_probs = None
st.set_page_config(
    page_title="Cardiovascular Disease Prediction Ensamble",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    font-family: Arial, sans-serif;
}
.stExpander {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    margin-bottom: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
h1, h4, p, .stMarkdown, .caption-text {
    color: #333333;
}
.bold-text {
    font-weight: bold;
    font-size: 1.2em;
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; text-align: center;">
    <h1>Cardiovascular Disease Prediction Using Ensemble Technique</h1>
</div>
""", unsafe_allow_html=True)

ecg = ECG()

st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
    <h4>Upload your ECG image for analysis</h4>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    try:
        st.markdown("### Uploaded Image")
        ecg_user_image_read = ecg.getImage(uploaded_file)
        st.image(ecg_user_image_read, caption="", use_column_width=True)
        # Apply the same caption style
        st.markdown("<p style='color: black; font-size: 16px; font-weight: bold; text-align: center;'>Uploaded ECG Image</p>", unsafe_allow_html=True)

        with st.expander("Gray Scale Image"):
            ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
            st.image(ecg_user_gray_image_read, caption="", use_column_width=True)
            # Apply the same caption style for grayscale image
            st.markdown("<p style='color: black; font-size: 16px; font-weight: bold; text-align: center;'>Grayscale ECG Image</p>", unsafe_allow_html=True)

        # Continue with other sections...


            
            

        with st.expander("Dividing Leads"):
            dividing_leads = ecg.DividingLeads(ecg_user_image_read)
            col1, col2 = st.columns(2)
            with col1:
                 st.image('Leads_1-12_figure.png', use_column_width=True)
                 st.markdown("""   <p style='color: black; font-size: 16px; font-weight: bold; text-align: center;'>Long Lead 1-12</p> """, unsafe_allow_html=True)
              
            with col2:
               st.image('Long_Lead_13_figure.png', use_column_width=True)
               st.markdown("""   <p style='color: black; font-size: 16px; font-weight: bold; text-align: center;'>Long Lead 13</p> """, unsafe_allow_html=True)
               
                 

# Use st.markdown to display the caption with HTML and custom styling
        with st.expander("Preprocessed Leads"):
          ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
          col1, col2 = st.columns(2)
          with col1:
            st.image('Preprossed_Leads_1-12_figure.png', caption="", use_column_width=True)
        # Apply caption style for Preprocessed Leads 1-12
            st.markdown("<p style='color: black; font-size: 16px; font-weight: bold; text-align: center;'>Preprocessed Leads 1-12</p>", unsafe_allow_html=True)
          with col2:
            st.image('Preprossed_Leads_13_figure.png', caption="", use_column_width=True)
        # Apply caption style for Preprocessed Lead 13
            st.markdown("<p style='color: black; font-size: 16px; font-weight: bold; text-align: center;'>Preprocessed Lead 13</p>", unsafe_allow_html=True)

        with st.expander("Extracting Signals"):
          ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
          st.image('Contour_Leads_1-12_figure.png', caption="", use_column_width=True)
    # Apply caption style for Signal Contours
          st.markdown("<p style='color: black; font-size: 16px; font-weight: bold; text-align: center;'>Signal Contours</p>", unsafe_allow_html=True)




        with st.expander("1D Signal Conversion"):
            ecg_1dsignal = ecg.CombineConvert1Dsignal()
            st.write(ecg_1dsignal)

        with st.expander("Dimensionality Reduction"):
            ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
            st.write(ecg_final)

      

        with st.expander("Prediction Results Using Soft Voting method (averaging)"):
    # Perform prediction and get both diagnosis, probabilities for each base learner, and ensemble probabilities
            diagnosis, base_learner_probs, ensemble_probs = ecg.ModelLoad_predict_soft_voting(ecg_final)

            # Display the diagnosis in bold text
            st.markdown(f"<p class='bold-text'>Diagnosis: {diagnosis}</p>", unsafe_allow_html=True)

            # Create a mapping for the classes to ensure clarity
            class_mapping = {
                0: "Abnormal Heartbeat",
                1: "Myocardial Infarction",
                2: "Normal",
                3: "History of Myocardial Infarction"
            }

            # Display ensemble model prediction probabilities
            st.subheader("Ensemble Model Prediction Probabilities:")
            for i, class_name in class_mapping.items():
                st.write(f"{class_name}: {ensemble_probs[i]:.4f}")

            # Display base learners' probabilities
            st.subheader("Base Learners' Probabilities:")

            # Unpack probabilities for each base learner from the dictionary
            svm_probs = base_learner_probs['SVM']
            knn_probs = base_learner_probs['KNN']
            rf_probs = base_learner_probs['RF']
            bayes_probs = base_learner_probs['Bayes']
            logistic_probs = base_learner_probs['Logistic']

            # Display the probabilities for SVM
            st.write("**SVM (Support Vector Machine)**")
            for i, class_name in class_mapping.items():
                st.write(f"{class_name}: {svm_probs[i]:.4f}")

            # Display the probabilities for KNN
            st.write("**KNN (K-Nearest Neighbors)**")
            for i, class_name in class_mapping.items():
                st.write(f"{class_name}: {knn_probs[i]:.4f}")

            # Display the probabilities for Random Forest
            st.write("**Random Forest**")
            for i, class_name in class_mapping.items():
                st.write(f"{class_name}: {rf_probs[i]:.4f}")

            # Display the probabilities for Naive Bayes
            st.write("**Naive Bayes**")
            for i, class_name in class_mapping.items():
                st.write(f"{class_name}: {bayes_probs[i]:.4f}")

            # Display the probabilities for Logistic Regression
            st.write("**Logistic Regression**")
            for i, class_name in class_mapping.items():
                st.write(f"{class_name}: {logistic_probs[i]:.4f}")
        


        with st.expander("Prediction Results Using Hard Voting Method"):
    # Perform prediction and get both diagnosis and voting arrays
            diagnosis, voting_arrays = ecg.ModelLoad_predict_hard_voting(ecg_final, svm_probs, knn_probs, rf_probs, bayes_probs, logistic_probs)
            
            # Display the diagnosis
            st.markdown(f"<p class='bold-text'>Diagnosis: {diagnosis}</p>", unsafe_allow_html=True)
            
            # Display base learner voting arrays
            st.write("Base Learner Prediction Voting Arrays (0-1):")
            
            # Display the 0-1 voting arrays for each base learner
            st.write("SVM Voting Array:", voting_arrays['SVM'])
            st.write("KNN Voting Array:", voting_arrays['KNN'])
            st.write("RF Voting Array:", voting_arrays['RF'])
            st.write("Bayes Voting Array:", voting_arrays['Bayes'])
            st.write("Logistic Voting Array:", voting_arrays['Logistic'])
            
            
            # Calculate the final class prediction based on majority voting
            # Count how many times each class (0, 1, 2, 3) has been voted
            all_votes = [voting_arrays['SVM'], voting_arrays['KNN'], voting_arrays['RF'], voting_arrays['Bayes'], voting_arrays['Logistic']]
            
            # Sum the votes for each class (index 0 to 3)
            vote_sums = [sum(vote[i] for vote in all_votes) for i in range(4)]
            
            # Determine the class with the most votes (highest sum)
            final_class = np.argmax(vote_sums)
            
            # Final diagnosis based on majority voting
            if final_class == 1:
                diagnosis = "Your ECG corresponds to Myocardial Infarction"
            elif final_class == 0:
                diagnosis = "Your ECG corresponds to Abnormal Heartbeat"
            elif final_class == 2:
                diagnosis = "Your ECG is Normal"
            else:
                diagnosis = "Your ECG corresponds to History of Myocardial Infarction"
            
            # Display final prediction and the votes that led to it
            st.write(f"Final Class Prediction (based on hard voting): {diagnosis}")
            st.write(f"Votes for the Final Prediction: {vote_sums[final_class]}")






       


    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

st.markdown("""
<div style="text-align: center; margin-top: 50px;">
    <p>© 2024 Cardiovascular Disease Prediction Using Ensemble Technique</p>
</div>
""", unsafe_allow_html=True)
