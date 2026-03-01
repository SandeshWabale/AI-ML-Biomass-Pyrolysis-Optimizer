document.getElementById('optimize-button').addEventListener('click', function(event) {
    event.preventDefault();
    runOptimization();
});

function getFormData() {
   
    const getFormValues = (formId) => {
        const data = {};
        const form = document.getElementById(formId);
        Array.from(form.elements).forEach(el => {
            if (el.id && el.type === 'number') data[el.id] = parseFloat(el.value);
        });
        return data;
    };
    
    
    const constraintsData = {
        PS: [parseFloat(document.getElementById('min_PS').value), parseFloat(document.getElementById('max_PS').value)],
        FT: [parseFloat(document.getElementById('min_FT').value), parseFloat(document.getElementById('max_FT').value)],
        HR: [parseFloat(document.getElementById('min_HR').value), parseFloat(document.getElementById('max_HR').value)],
        FR: [parseFloat(document.getElementById('min_FR').value), parseFloat(document.getElementById('max_FR').value)],
    };

    
    return {
        biomass_properties: getFormValues('biomass-form'),
        current_process_params: getFormValues('params-form'),
        optimization_goal: document.getElementById('goal-yield').value,
        constraints: constraintsData
    };
}

async function runOptimization() {
    const data = getFormData();
    const resultsOutput = document.querySelector('#results-output .results-table-container');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    resultsOutput.innerHTML = '<p>Running advanced optimization... this may take a moment.</p>';
    loadingSpinner.classList.remove('hidden');

    try {
        const response = await fetch('http://127.0.0.1:8000/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'The server returned an error.');
        }

        const result = await response.json();
        displayResults(result, data.current_process_params);

    } catch (error) {
        console.error("Frontend Error:", error);
        resultsOutput.innerHTML = `<p style="color: red;"><strong>Error:</strong> ${error.message}<br>Please ensure the Python server is running and check the browser's developer console (F12) for more details.</p>`;
    } finally {
        loadingSpinner.classList.add('hidden');
    }
}

function displayResults(result, currentParams) {
    const resultsContainer = document.querySelector('#results-output .results-table-container');
    const goal = result.optimization_goal;
    
    let html = `<table class="results-table"><thead><tr><th>Metric</th><th>Current Value</th><th>Current Yield</th><th>Optimized Value</th><th>Optimized Yield</th></tr></thead><tbody>`;
    const yieldPhases = ['Solid phase', 'Liquid phase', 'Gas phase'];
    yieldPhases.forEach(phase => {
        const isGoal = phase === goal;
        const currentYield = result.current_yields[phase].toFixed(2);
        const optimizedYield = result.optimized_yields[phase].toFixed(2);
        html += `<tr class="${isGoal ? 'highlight' : ''}"><td>${phase} (%)</td><td>-</td><td>${currentYield}</td><td>-</td><td>${optimizedYield}</td></tr>`;
    });

    const paramNames = { 'PS': 'Particle Size (mm)', 'FT': 'Final Temp (°C)', 'HR': 'Heating Rate (°C/min)', 'FR': 'Flow Rate (mL/min)' };
    
    ['PS', 'FT', 'HR', 'FR'].forEach(param => {
        const optimizedVal = result.optimized_params[param].toFixed(2);
        const currentVal = currentParams[param].toFixed(2);
        html += `<tr><td>${paramNames[param]}</td><td>${currentVal}</td><td>-</td><td class="${Math.abs(optimizedVal - currentVal) > 0.05 ? 'highlight-param' : ''}">${optimizedVal}</td><td>-</td></tr>`;
    });

    html += `</tbody></table>`;
    resultsContainer.innerHTML = html;
}