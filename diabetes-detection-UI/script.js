const form = document.getElementById('prediction-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric fields to numbers
    for (let key in data) {
        data[key] = parseFloat(data[key]);
    }

    try {
       const response = await fetch('https://diabetes-detection-api-xnu6.onrender.com/predict', {

            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        resultDiv.innerHTML = `
            Prediction: <strong>${result.result}</strong><br>
            Probability: <strong>${(result.probability * 100).toFixed(2)}%</strong>
        `;
    } catch (error) {
        resultDiv.innerHTML = 'Error connecting to API.';
        console.error(error);
    }
});
