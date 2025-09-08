
import pandas as pd
import numpy as np
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CropRecommendationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Crop Recommendation System")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')

        self.load_model()

        self.create_widgets()

    def load_model(self):
        """Load the pre-trained model and components"""
        try:
            with open('crop_recommendation_model.pkl', 'rb') as f:
                components = pickle.load(f)

            self.model = components['model']
            self.scaler = components['scaler'] 
            self.crop_db = components['crop_database']
            self.feature_names = components['feature_names']
            self.accuracy = components['accuracy']

            print(f"Model loaded successfully! Accuracy: {self.accuracy:.4f}")

        except FileNotFoundError:
            messagebox.showerror("Error", "Model file not found. Please train the model first.")

    def create_widgets(self):
        """Create the GUI interface"""
        
        title_label = tk.Label(
            self.root, 
            text="🌾 Smart Crop Recommendation System 🌾",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2e7d32'
        )
        title_label.pack(pady=20)

        
        subtitle_label = tk.Label(
            self.root,
            text="Enter your soil and weather conditions to get crop recommendations",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#666666'
        )
        subtitle_label.pack(pady=(0, 20))

        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)

        input_frame = tk.LabelFrame(main_frame, text="Soil and Weather Conditions", 
                                   font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2e7d32')
        input_frame.pack(fill='x', pady=(0, 20))

        self.create_input_fields(input_frame)

        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=10)

        predict_btn = tk.Button(
            button_frame,
            text="🔍 Get Crop Recommendations",
            command=self.predict_crop,
            font=("Arial", 12, "bold"),
            bg='#4caf50',
            fg='white',
            relief='flat',
            padx=30,
            pady=10
        )
        predict_btn.pack(side='left', padx=(0, 10))

        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear All",
            command=self.clear_fields,
            font=("Arial", 12, "bold"),
            bg='#ff9800',
            fg='white',
            relief='flat',
            padx=30,
            pady=10
        )
        clear_btn.pack(side='left')

        results_frame = tk.LabelFrame(main_frame, text="Crop Recommendations", 
                                     font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2e7d32')
        results_frame.pack(fill='both', expand=True, pady=(20, 0))

        self.results_text = tk.Text(
            results_frame, 
            height=15, 
            width=80, 
            font=("Courier", 10),
            bg='white',
            fg='#333333',
            relief='flat',
            padx=10,
            pady=10
        )

        scrollbar = tk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def create_input_fields(self, parent):
        """Create input fields for soil and weather parameters"""
        self.entries = {}

        fields = [
            ("Soil pH", "soil_ph", "6.5"),
            ("Temperature (°C)", "temperature", "25"),
            ("Rainfall (mm)", "rainfall", "800"),
            ("Nitrogen (kg/ha)", "nitrogen", "120"),
            ("Phosphorus (kg/ha)", "phosphorus", "60"),
            ("Potassium (kg/ha)", "potassium", "60"),
            ("Humidity (%)", "humidity", "70"),
            ("Current Month (1-12)", "month", str(datetime.now().month))
        ]

        for i, (label, key, default) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2

            tk.Label(
                parent, 
                text=label + ":", 
                font=("Arial", 10),
                bg='#f0f0f0'
            ).grid(row=row, column=col, sticky='e', padx=(10, 5), pady=8)

            entry = tk.Entry(
                parent, 
                font=("Arial", 10),
                width=15,
                relief='flat',
                bd=1
            )
            entry.insert(0, default)
            entry.grid(row=row, column=col+1, sticky='w', padx=(5, 20), pady=8)

            self.entries[key] = entry

    def get_season_from_month(self, month):
        """Convert month to season"""
        if month in [6, 7, 8, 9]:
            return 1  # Kharif
        elif month in [10, 11, 12, 1]:
            return 2  # Rabi
        elif month in [3, 4, 5]:
            return 3  # Zaid
        else:
            return 2  # Default to Rabi

    def determine_soil_type(self, soil_ph, nitrogen, phosphorus, potassium):
        """Determine soil type based on nutrient content"""
        if nitrogen > 150 and phosphorus > 80 and potassium > 100:
            return 4  # Alluvial (very fertile)
        elif nitrogen > 100 and phosphorus > 60:
            return 5  # Black soil (cotton belt)
        elif nitrogen < 80 and phosphorus < 40:
            return 1  # Sandy (low nutrients)
        elif soil_ph > 7.0:
            return 3  # Clay (alkaline)
        else:
            return 2  # Loamy (balanced)

    def predict_crop(self):
        """Make crop prediction based on input values"""
        try:
            soil_ph = float(self.entries["soil_ph"].get())
            temperature = float(self.entries["temperature"].get())
            rainfall = float(self.entries["rainfall"].get())
            nitrogen = float(self.entries["nitrogen"].get())
            phosphorus = float(self.entries["phosphorus"].get())
            potassium = float(self.entries["potassium"].get())
            humidity = float(self.entries["humidity"].get())
            month = int(self.entries["month"].get())

            if not (1 <= month <= 12):
                raise ValueError("Month must be between 1 and 12")
            if not (0 <= soil_ph <= 14):
                raise ValueError("Soil pH must be between 0 and 14")
            if not (0 <= humidity <= 100):
                raise ValueError("Humidity must be between 0 and 100")

            season = self.get_season_from_month(month)
            soil_type = self.determine_soil_type(soil_ph, nitrogen, phosphorus, potassium)

            input_data = np.array([[
                soil_ph, temperature, rainfall, nitrogen, phosphorus, 
                potassium, humidity, month, season, soil_type
            ]])

            probabilities = self.model.predict_proba(input_data)[0]
            crop_names = self.model.classes_

            recommendations = []
            for i, crop in enumerate(crop_names):
                recommendations.append({
                    'crop': crop,
                    'confidence': probabilities[i],
                    'suitability_score': probabilities[i] * 100
                })

            recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)

            self.display_results(recommendations[:5], {
                'soil_ph': soil_ph,
                'temperature': temperature,
                'rainfall': rainfall,
                'season': season,
                'soil_type': soil_type,
                'month': month
            })

        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid values: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def display_results(self, recommendations, input_analysis):
        """Display the crop recommendations"""
        self.results_text.delete(1.0, tk.END)

        self.results_text.insert(tk.END, "🌾 CROP RECOMMENDATION RESULTS 🌾\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")

        season_names = {1: "Kharif", 2: "Rabi", 3: "Zaid"}
        soil_types = {1: "Sandy", 2: "Loamy", 3: "Clay", 4: "Alluvial", 5: "Black"}

        self.results_text.insert(tk.END, "📊 INPUT ANALYSIS:\n")
        self.results_text.insert(tk.END, f"• Season: {season_names.get(input_analysis['season'], 'Unknown')}\n")
        self.results_text.insert(tk.END, f"• Soil Type: {soil_types.get(input_analysis['soil_type'], 'Unknown')}\n")
        self.results_text.insert(tk.END, f"• Soil pH: {input_analysis['soil_ph']:.1f}\n")
        self.results_text.insert(tk.END, f"• Temperature: {input_analysis['temperature']}°C\n")
        self.results_text.insert(tk.END, f"• Rainfall: {input_analysis['rainfall']}mm\n\n")

        self.results_text.insert(tk.END, "🏆 TOP CROP RECOMMENDATIONS:\n")
        self.results_text.insert(tk.END, "-"*40 + "\n\n")

        for i, rec in enumerate(recommendations, 1):
            crop_info = self.crop_db[self.crop_db['crop_name'] == rec['crop']]

            if not crop_info.empty:
                crop_info = crop_info.iloc[0]

                self.results_text.insert(tk.END, f"{i}. {rec['crop'].replace('_', ' ').upper()}\n")
                self.results_text.insert(tk.END, f"   🎯 Suitability Score: {rec['suitability_score']:.1f}%\n")
                self.results_text.insert(tk.END, f"   📈 Expected Yield: {crop_info['expected_yield']} quintals/ha\n")
                self.results_text.insert(tk.END, f"   ⏱️ Crop Duration: {crop_info['crop_duration']} days\n")

                if rec['suitability_score'] >= 70:
                    self.results_text.insert(tk.END, "   ✅ Highly Recommended\n")
                elif rec['suitability_score'] >= 40:
                    self.results_text.insert(tk.END, "   ⚠️ Moderately Suitable\n")
                else:
                    self.results_text.insert(tk.END, "   ❌ Not Recommended\n")

                self.results_text.insert(tk.END, "\n")

        self.results_text.insert(tk.END, "="*60 + "\n")
        self.results_text.insert(tk.END, f"Model Accuracy: {self.accuracy:.1%}\n")
        self.results_text.insert(tk.END, "💡 Tip: Consider local market prices and farming expertise!")

    def clear_fields(self):
        """Clear all input fields"""
        defaults = {
            "soil_ph": "6.5",
            "temperature": "25", 
            "rainfall": "800",
            "nitrogen": "120",
            "phosphorus": "60",
            "potassium": "60",
            "humidity": "70",
            "month": str(datetime.now().month)
        }

        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, defaults[key])

        self.results_text.delete(1.0, tk.END)

def main():
    root = tk.Tk()
    app = CropRecommendationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
