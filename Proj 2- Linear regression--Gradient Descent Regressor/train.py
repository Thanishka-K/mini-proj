import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import LinearRegressor

def main():
    data = pd.read_csv("linreg_data.csv")
    X = data[['X']].values 
    y = data['y'].values

    # Initialize and Train Model
    # Experiment with learning_rate (alpha) to see how it affects convergence!
    model = LinearRegressor(learning_rate=0.01, iterations=1000)
    
    print("🚀 Training starting...")
    model.fit(X, y)
    print("✅ Training complete!")

    print(f"\nFinal Weight (m): {model.weights[0]:.4f}") # Parameters
    print(f"Final Bias (b): {model.bias:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')

    y_pred = model.predict(X)
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line (Scratch)')
    
    plt.xlabel('X (Input Feature)')
    plt.ylabel('y (Target Variable)')
    plt.title('Linear Regression from Scratch: Results')
    plt.legend()

    plt.savefig("regression_result.png")
    print("\n[SUCCESS] Result plot saved as 'regression_result.png'")
    plt.show()

if __name__ == "__main__":
    main()
