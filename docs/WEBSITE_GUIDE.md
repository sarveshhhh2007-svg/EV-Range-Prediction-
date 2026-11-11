# Website Guide: EV Range Predictor

A walkthrough of the 5-section web application explaining every item in plain language.

---

## 1. Top Navigation Bar

### Project Brand & Icon
- **What you see:** A burgundy bolt icon and the text "EV Range Predictor".
- **What it means:** The application identity. Clicking it returns to the top.
- **Why it is here:** Provides clear branding and quick return navigation.

### Navigation Links (Home, Data, Insights, Models, Predict)
- **What you see:** Five menu links across the top bar.
- **What it means:** Shortcuts to each of the five sections on the page.
- **Why it is here:** Lets visitors jump smoothly between key parts of the project.

### Mobile Menu Button
- **What you see:** A menu toggle icon on mobile screens.
- **What it means:** Expands or collapses the five navigation links.
- **Why it is here:** Keeps the navigation clean and accessible on phones.

---

## 2. Section 1: Home

### Hero Title
- **What you see:** "Comparative Analysis of Machine Learning Regression Models for Electric Vehicle Driving-Range Prediction".
- **What it means:** The official project title describing the study.
- **Why it is here:** States the exact academic scope of the work upfront.

### Hero Description & Scope Note
- **What you see:** A short description and a boxed scope note.
- **What it means:** The model predicts listed catalog driving range from vehicle specifications.
- **Why it is here:** Clarifies that predictions reflect official ratings, not battery degradation or real-world weather tests.

### "Predict Range" Button
- **What you see:** A burgundy action button.
- **What it means:** A direct link that scrolls straight to the interactive prediction form.
- **Why it is here:** Allows quick access to the interactive prediction tool.

### Three Summary Bars (Cars, Specs, Models)
- **What you see:** Three vertical metric bars showing 478 cars, 18 specs, and 3 models.
- **What it means:** The total volume of real vehicles analysed, inputs considered, and models trained.
- **Why it is here:** Gives viewers an immediate understanding of the project's scale.

### 4-Step Process Strip
- **What you see:** Four numbered connected steps: Collect data → Clean data → Train 3 models → Predict range.
- **What it means:** The full path taken from raw data to the live prediction tool.
- **Why it is here:** Summarises the overall workflow at a glance.

### Best Model & SDG Banner
- **What you see:** A summary banner naming Linear Regression with 14.37 km average error, and an SDG 9 badge.
- **What it means:** Identifies the top-performing model and links the study to UN Sustainable Development Goal 9.
- **Why it is here:** Highlights our key finding and real-world technology relevance.

---

## 3. Section 2: Vehicle Data

### Search Bar & Car Counter
- **What you see:** A text box and a counter showing "478 cars".
- **What it means:** Filters the table in real time as you type a brand or model name.
- **Why it is here:** Makes it easy to search specific cars across the dataset.

### 6-Column Vehicle Table
- **What you see:** A table showing Brand, Model, Battery (kWh), Top Speed (km/h), 0–100 km/h (s), and Range (km).
- **What it means:** Standardised technical specifications for 10 cars per page.
- **Why it is here:** Lets visitors inspect the real car data used in this study.

### Pagination Controls
- **What you see:** "Previous" and "Next" buttons with a page counter.
- **What it means:** Moves through the 48 pages of vehicles.
- **Why it is here:** Keeps page loading fast and browsing clean.

### "How We Cleaned the Data" Bullets
- **What you see:** Six plain-language bullet points.
- **What it means:** Explains how missing values, non-numeric text, and test splits were handled.
- **Why it is here:** Shows transparency on data preparation without confusing mathematical jargon.

---

## 4. Section 3: Data Insights

### Chart 1: Driving Range Distribution
- **What you see:** A bar chart showing how many cars fall into each 50 km range bucket.
- **What it means:** Most electric vehicles in 2025 offer between 350 km and 450 km of range.
- **Why it is here:** Shows the overall spread and typical range of modern EVs.

### Chart 2: Battery Capacity vs Range
- **What you see:** A scatter plot with battery capacity along the bottom and range on the vertical axis.
- **What it means:** Vehicles equipped with larger batteries travel further on a single charge.
- **Why it is here:** Confirms battery size as the single strongest factor in driving range.

### Chart 3: Acceleration Time vs Range
- **What you see:** A scatter plot with 0–100 km/h time along the bottom and range on the vertical axis.
- **What it means:** Cars that accelerate faster often carry larger batteries and deliver longer range.
- **Why it is here:** Shows how vehicle performance tiers relate to driving range.

### Chart 4: What Moves with Range (Correlations)
- **What you see:** A horizontal bar chart displaying 8 car specifications.
- **What it means:** Longer bars indicate specs that strongly rise or fall alongside driving range.
- **Why it is here:** Identifies which technical specifications matter most when estimating range.

---

## 5. Section 4: Model Performance

### Three Model Summary Cards
- **What you see:** Three cards for Linear Regression, Random Forest, and Gradient Boosting.
- **What it means:** Brief 1-sentence explanations of how each algorithm approaches predictions.
- **Why it is here:** Helps non-technical visitors understand the difference between the 3 models.

### Results Metrics Table
- **What you see:** A table listing Average error (MAE), Squared error (MSE), Big-miss error (RMSE), and Fit score (R²).
- **What it means:** Objective test scores measured across 96 unseen vehicles.
- **Why it is here:** Compares accuracy across all three models with clear metric labels.

### Chart 5: Average Error Comparison (MAE)
- **What you see:** Three vertical bars comparing the average error of each model in kilometres.
- **What it means:** Shorter bars mean the model's estimates are closer to actual listed numbers.
- **Why it is here:** Provides an instant visual benchmark of model accuracy.

### Chart 6: Actual vs Predicted Scatter
- **What you see:** A scatter chart with a diagonal dashed line for the best model.
- **What it means:** Dots lying close to the line represent accurate predictions on unseen test cars.
- **Why it is here:** Proves the model makes dependable estimates across the entire range span.

### Best Model Callout & Honest Note
- **What you see:** A callout naming Linear Regression as best, with an honest note below.
- **What it means:** Explains why it was selected and notes that closely scoring models are equally capable.
- **Why it is here:** Communicates academic rigor and realistic machine learning evaluation.

---

## 6. Section 5: Predict EV Driving Range

### Main 6 Inputs
- **What you see:** Six input boxes for Battery, Top Speed, Acceleration, Fast-charging, Torque, and Efficiency.
- **What it means:** The six strongest technical factors needed to calculate driving range.
- **Why it is here:** Keeps the prediction form fast, simple, and intuitive to use.

### "More Details (Optional)" Collapsed Area
- **What you see:** An expandable tray containing secondary specs (seats, dimensions, brand, body type).
- **What it means:** Optional specifications that can be filled to refine the prediction.
- **Why it is here:** Missing fields are automatically filled; entering them allows closer estimates.

### "Try an Example" Button
- **What you see:** A text button next to the main predict button.
- **What it means:** Automatically fills the form with a real unseen test car and runs the prediction.
- **Why it is here:** Lets visitors test the tool with one click without typing numbers manually.

### "Clear" Button
- **What you see:** A reset button.
- **What it means:** Empties all form inputs and hides prediction results.
- **Why it is here:** Lets users quickly start over with a fresh calculation.

### "Predict EV Driving Range" Button
- **What you see:** A large burgundy button.
- **What it means:** Submits entered specs to all three trained models simultaneously.
- **Why it is here:** Triggers the machine learning prediction calculation.

### Unusual Input Warning
- **What you see:** A soft peach callout warning box.
- **What it means:** Alerts the user if entered numbers are unusually far outside the training dataset.
- **Why it is here:** Promotes safe usage and prevents misleading claims on extreme values.

### Result Area & CSS Comparison Bars
- **What you see:** The best model's predicted range in large text, followed by 3 horizontal progress bars.
- **What it means:** Shows the predicted range from all three models side by side.
- **Why it is here:** Visualises agreement and differences among the different algorithms.

---

## 7. Footer

### Project Title & Scope Notice
- **What you see:** "EV Range Predictor" and "Estimates are based on listed specifications, not real-world driving."
- **What it means:** A simple concluding statement on the tool's intended use.
- **Why it is here:** Maintains clear scope without distracting student or course metadata.
