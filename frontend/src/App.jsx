import { useCallback, useMemo, useState } from "react";

const API_URL = "http://localhost:8001/predict";

const CATEGORY_INFO = {
  Lifestyle: "Smoking, alcohol, BMI-related data, activity, and diet profile",
  Genetic: "Family history and inherited or polygenic risk",
  Environmental: "Pollution, UV/radiation, and occupational hazard exposure",
  Clinical: "Core demographics and comorbidity context",
};

const CLINICAL_SYMPTOMS = [
  {
    id: "unexplained_weight_loss",
    label: "Unexplained Weight Loss: Rapidly losing weight without dieting or extra exercise.",
  },
  {
    id: "new_lumps",
    label: "New Lumps or Hard Spots: Any new, firm growth or bump felt under the skin.",
  },
  {
    id: "unusual_bleeding",
    label: "Unusual Bleeding: Blood in spit, urine, stool, or unusual bruising.",
  },
  {
    id: "non_healing_sores",
    label: "Non-Healing Sores: A wound or sore that does not heal after several weeks.",
  },
  {
    id: "changed_bathroom_habits",
    label: "Changed Bathroom Habits: Long-term bowel/urine changes or constant stomach pain.",
  },
  {
    id: "persistent_cough",
    label: "Persistent Cough: Cough or hoarseness lasting more than 3 weeks.",
  },
  {
    id: "changing_moles",
    label: "Changing Moles: Skin mark changing color, size, or shape.",
  },
  {
    id: "difficulty_swallowing",
    label: "Difficulty Swallowing: Food feeling stuck or pain while swallowing.",
  },
];

const steps = [
  {
    key: "smoke",
    label: "How often do you smoke?",
    type: "options",
    options: [
      { label: "No", value: 0 },
      { label: "Occasionally", value: 1 },
      { label: "Weekly", value: 2 },
      { label: "Daily", value: 3 },
    ],
    category: "Lifestyle",
  },
  {
    key: "alcohol_intake",
    label: "Alcohol intake range (0 to 5)?",
    type: "options",
    options: [
      { label: "Never drink", value: 0 },
      { label: "1-3 times a month", value: 1.5 },
      { label: "1-3 times a week", value: 2.5 },
      { label: "4-6 times a week", value: 3.7 },
      { label: "Daily or almost daily", value: 5 },
    ],
    category: "Lifestyle",
  },
  {
    key: "physical_activity",
    label: "How often are you physically active?",
    type: "options",
    options: [
      { label: "No regular activity", value: 0 },
      { label: "1-2 times a month", value: 1.5 },
      { label: "1-2 times a week", value: 2.5 },
      { label: "3-5 times a week", value: 3.7 },
      { label: "Daily", value: 5 },
    ],
    category: "Lifestyle",
  },
  {
    key: "height_cm",
    label: "What is your height in cm?",
    type: "number",
    min: 120,
    max: 230,
    step: 1,
    placeholder: "Example: 170",
    category: "Lifestyle",
  },
  {
    key: "weight_kg",
    label: "What is your weight in kg?",
    type: "number",
    min: 30,
    max: 250,
    step: 0.1,
    placeholder: "Example: 68.5",
    category: "Lifestyle",
  },
  {
    key: "diet_pattern",
    label: "Diet pattern over the past year?",
    type: "options",
    options: [
      { label: "Mostly plant-based", value: 0 },
      { label: "Mixed diet", value: 1 },
      { label: "High red/processed meat", value: 2 },
    ],
    category: "Lifestyle",
  },
  {
    key: "genetic_risk",
    label: "Genetic risk level based on family genetics?",
    type: "options",
    options: [
      { label: "No", value: 0 },
      { label: "Maybe", value: 1 },
      { label: "Yes", value: 2 },
    ],
    category: "Genetic",
  },
  {
    key: "early_onset",
    label: "Did any family members get cancer at a young age (before 50)?",
    type: "options",
    options: [
      { label: "Yes", value: 2 },
      { label: "No", value: 0 },
      { label: "Not Known", value: 1 },
    ],
    category: "Genetic",
  },
  {
    key: "known_gene",
    label: "Has a doctor ever told you that you have a cancer gene (like BRCA)?",
    type: "options",
    options: [
      { label: "Yes", value: 2 },
      { label: "No", value: 0 },
      { label: "Not Known", value: 1 },
    ],
    category: "Genetic",
  },
  {
    key: "family_history",
    label: "Did anyone in your family have cancer?",
    type: "yesno",
    category: "Genetic",
  },
  {
    key: "pollution_exposure",
    label: "Do you have regular high pollution exposure?",
    type: "yesno",
    category: "Environmental",
  },
  {
    key: "uv_radiation_exposure",
    label: "Do you have high UV or radiation exposure?",
    type: "yesno",
    category: "Environmental",
  },
  {
    key: "occupational_hazard",
    label: "Any occupational hazard exposure (chemicals/asbestos)?",
    type: "yesno",
    category: "Environmental",
  },
  {
    key: "age",
    label: "What is your age?",
    type: "number",
    min: 20,
    max: 80,
    step: 1,
    placeholder: "Enter your age (20-80)",
    category: "Clinical",
  },
  {
    key: "gender",
    label: "What is your gender?",
    type: "options",
    options: [
      { label: "Female", value: 0 },
      { label: "Male", value: 1 },
    ],
    category: "Clinical",
  },
  {
    key: "diabetes",
    label: "Do you have diabetes?",
    type: "yesno",
    category: "Clinical",
  },
  {
    key: "bp_level",
    label: "Do you have BP (high blood pressure)?",
    type: "yesno",
    category: "Clinical",
  },
  {
    key: "fatigue_level",
    label: "How often do you feel unusual fatigue?",
    type: "options",
    options: [
      { label: "Rarely", value: 0 },
      { label: "Sometimes", value: 1 },
      { label: "Often", value: 2 },
      { label: "Almost daily", value: 3 },
    ],
    category: "Clinical",
  },
  {
    key: "symptoms",
    label: "Are you currently facing any of these symptoms? (Tick all that apply)",
    type: "multicheck",
    options: CLINICAL_SYMPTOMS,
    category: "Clinical",
  },
];

function App() {
  const [categoryIndex, setCategoryIndex] = useState(0);
  const [answers, setAnswers] = useState({
    age: "",
    gender: null,
    height_cm: "",
    weight_kg: "",
    smoke: null,
    diet_pattern: null,
    genetic_risk: null,
    early_onset: null,
    known_gene: null,
    family_history: null,
    physical_activity: null,
    alcohol_intake: null,
    pollution_exposure: null,
    uv_radiation_exposure: null,
    occupational_hazard: null,
    diabetes: null,
    bp_level: null,
    fatigue_level: null,
    symptoms: [],
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const categories = useMemo(() => [...new Set(steps.map((step) => step.category))], []);
  const categoryQuestions = useMemo(
    () => categories.map((category) => ({ category, questions: steps.filter((step) => step.category === category) })),
    [categories]
  );

  const currentCategoryBlock = categoryQuestions[categoryIndex];
  const currentCategory = currentCategoryBlock?.category;
  const currentQuestions = currentCategoryBlock?.questions || [];
  const isLastCategory = categoryIndex === categoryQuestions.length - 1;

  const isQuestionAnswered = useCallback((question) => {
    const value = answers[question.key];
    if (question.type === "number") {
      if (value === "") {
        return false;
      }
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return false;
      }
      return numeric >= question.min && numeric <= question.max;
    }

    if (question.type === "multicheck") {
      return true;
    }

    return value !== null && value !== undefined;
  }, [answers]);

  const calculatedBmi = useMemo(() => {
    const heightCm = Number(answers.height_cm);
    const weightKg = Number(answers.weight_kg);

    if (!Number.isFinite(heightCm) || !Number.isFinite(weightKg) || heightCm <= 0 || weightKg <= 0) {
      return null;
    }

    const heightM = heightCm / 100;
    return weightKg / (heightM * heightM);
  }, [answers.height_cm, answers.weight_kg]);

  const incompleteQuestions = useMemo(
    () => currentQuestions.filter((question) => !isQuestionAnswered(question)),
    [isQuestionAnswered, currentQuestions]
  );

  const canContinue = incompleteQuestions.length === 0;

  const isGeneticsFollowupLocked = (questionKey) => {
    if (questionKey !== "early_onset" && questionKey !== "known_gene") {
      return false;
    }
    return answers.genetic_risk === 0 || answers.genetic_risk === 1;
  };

  const handleOptionSelect = (question, optionValue) => {
    if (isGeneticsFollowupLocked(question.key)) {
      return;
    }

    if (question.key === "genetic_risk") {
      if (optionValue === 0) {
        setAnswers({
          ...answers,
          genetic_risk: optionValue,
          early_onset: 0,
          known_gene: 0,
        });
        return;
      }

      if (optionValue === 1) {
        setAnswers({
          ...answers,
          genetic_risk: optionValue,
          early_onset: 1,
          known_gene: 1,
        });
        return;
      }

      setAnswers({
        ...answers,
        genetic_risk: optionValue,
        early_onset: null,
        known_gene: null,
      });
      return;
    }

    setAnswers({ ...answers, [question.key]: optionValue });
  };

  const toggleSymptom = (symptomId) => {
    const selected = answers.symptoms || [];
    const next = selected.includes(symptomId)
      ? selected.filter((id) => id !== symptomId)
      : [...selected, symptomId];
    setAnswers({ ...answers, symptoms: next });
  };

  const renderQuestionField = (question) => {
    if (question.type === "number") {
      return (
        <>
          <input
            className="input"
            type="number"
            min={question.min}
            max={question.max}
            step={question.step ?? 1}
            value={answers[question.key]}
            onChange={(e) => setAnswers({ ...answers, [question.key]: e.target.value })}
            onKeyDown={handleEnterKey}
            placeholder={question.placeholder || "Enter value"}
          />
          {question.key === "weight_kg" && calculatedBmi !== null && (
            <p className="helper-text">Calculated BMI: {calculatedBmi.toFixed(1)}</p>
          )}
        </>
      );
    }

    if (question.type === "multicheck") {
      const selected = answers[question.key] || [];
      return (
        <div className="symptom-checklist">
          {question.options?.map((option) => (
            <label key={option.id} className="symptom-item">
              <input
                type="checkbox"
                checked={selected.includes(option.id)}
                onChange={() => toggleSymptom(option.id)}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
      );
    }

    const options =
      question.type === "yesno"
        ? [
            { label: "Yes", value: 1 },
            { label: "No", value: 0 },
          ]
        : question.options;

    return (
      <div className="yes-no-grid">
        {options?.map((option) => (
          <button
            key={`${question.key}-${option.value}`}
            className={`option-btn ${answers[question.key] === option.value ? "selected" : ""}`}
            onClick={() => handleOptionSelect(question, option.value)}
            disabled={isGeneticsFollowupLocked(question.key)}
          >
            {option.label}
          </button>
        ))}
      </div>
    );
  };

  const handleNext = async () => {
    setError("");

    if (!canContinue) {
      setError(`Please complete this category. Missing: ${incompleteQuestions[0]?.label}`);
      return;
    }

    if (!isLastCategory) {
      setCategoryIndex((prev) => prev + 1);
      return;
    }

    setLoading(true);
    try {
      const payload = {
        age: Number(answers.age),
        gender: Number(answers.gender),
        height_cm: Number(answers.height_cm),
        weight_kg: Number(answers.weight_kg),
        smoke: Number(answers.smoke) > 0 ? 1 : 0,
        genetic_risk: Number(answers.genetic_risk),
        early_onset: Number(answers.early_onset),
        known_gene: Number(answers.known_gene),
        family_history: Number(answers.family_history),
        physical_activity: Number(answers.physical_activity),
        alcohol_intake: Number(answers.alcohol_intake),
        diet_pattern: Number(answers.diet_pattern),
        pollution_exposure: Number(answers.pollution_exposure),
        uv_radiation_exposure: Number(answers.uv_radiation_exposure),
        occupational_hazard: Number(answers.occupational_hazard),
        diabetes: Number(answers.diabetes),
        bp_level: Number(answers.bp_level),
        fatigue_level: Number(answers.fatigue_level),
        symptom_flags: answers.symptoms,
      };

      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Could not get prediction. Please try again.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    setError("");
    if (categoryIndex > 0) {
      setCategoryIndex((prev) => prev - 1);
    }
  };

  const handleEnterKey = (event) => {
    if (event.key !== "Enter") {
      return;
    }

    event.preventDefault();
    if (!loading && canContinue) {
      handleNext();
    }
  };

  const resetAll = () => {
    setCategoryIndex(0);
    setResult(null);
    setError("");
    setAnswers({
      age: "",
      gender: null,
      height_cm: "",
      weight_kg: "",
      smoke: null,
      diet_pattern: null,
      genetic_risk: null,
      early_onset: null,
      known_gene: null,
      family_history: null,
      physical_activity: null,
      alcohol_intake: null,
      pollution_exposure: null,
      uv_radiation_exposure: null,
      occupational_hazard: null,
      diabetes: null,
      bp_level: null,
      fatigue_level: null,
      symptoms: [],
    });
  };

  const progressPercent = Math.round(((categoryIndex + 1) / categoryQuestions.length) * 100);

  if (result) {
    const isHighRisk = result.risk_level === "high";

    return (
      <main className={`page ${isHighRisk ? "risk-high" : "risk-low"}`}>
        <div className="bg-shape bg-shape-a" aria-hidden="true" />
        <div className="bg-shape bg-shape-b" aria-hidden="true" />
        <section className="card result-card">
          <div className="eyebrow">OncoPredict Result</div>
          <h1 className="title">Your Lifestyle Risk Snapshot</h1>
          <p className="warning">THIS IS NOT A MEDICAL DIAGNOSIS. CONSULT A DOCTOR.</p>

          <div className="score-panel">
            <p className="score-label">Risk Score</p>
            <p className="score-value">{Math.round(result.risk_score * 100)}%</p>
            <p className="score-message">{result.message}</p>
          </div>

          {isHighRisk ? (
            <>
              <h2 className="section-title">Recommended Support</h2>
              <div className="box-list">
                {result.doctors?.map((doctor) => (
                  <div key={doctor} className="box-item">
                    {doctor}
                  </div>
                ))}
              </div>

              <h2 className="section-title">What To Do Now</h2>
              <div className="box-list">
                {result.remedies?.map((tip) => (
                  <div key={tip} className="box-item">
                    {tip}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <>
              <h2 className="section-title">Healthy Momentum Plan</h2>
              <h3 className="sub-title">Keep Doing</h3>
              <div className="box-list">
                {result.maintenance?.map((tip) => (
                  <div key={tip} className="box-item">
                    {tip}
                  </div>
                ))}
              </div>

              <h3 className="sub-title">Improve More</h3>
              <div className="box-list">
                {result.improvements?.map((tip) => (
                  <div key={tip} className="box-item">
                    {tip}
                  </div>
                ))}
              </div>
            </>
          )}

          {result.specialist_recommendation && (
            <div className="specialist-alert">
              <span className="specialist-icon">⚠️</span>
              {result.specialist_recommendation}
            </div>
          )}

          <button className="primary-btn full-width" onClick={resetAll}>
            Check Again
          </button>
        </section>
      </main>
    );
  }

  return (
    <main className="page">
      <div className="bg-shape bg-shape-a" aria-hidden="true" />
      <div className="bg-shape bg-shape-b" aria-hidden="true" />
      <section className="card">
        <div className="eyebrow">OncoPredict</div>
        <h1 className="title">Cancer Risk Check</h1>
        <p className="tagline">A quick lifestyle screening experience</p>
        <p className="warning">THIS IS NOT A MEDICAL DIAGNOSIS. CONSULT A DOCTOR.</p>

        <div className="progress-wrap">
          <p className="progress">Category {categoryIndex + 1} of {categoryQuestions.length}</p>
          <div className="progress-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={progressPercent}>
            <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
          </div>
        </div>

        <div className="category-status-line" aria-label="Category progress">
          {categories.map((category, index) => {
            const status = index < categoryIndex ? "completed" : index === categoryIndex ? "current" : "pending";
            return (
              <div key={category} className="status-item-wrap">
                <div className={`status-item ${status}`}>
                  <span className="status-dot" />
                  <span>{category}</span>
                </div>
                {index < categories.length - 1 && (
                  <div className={`status-connector ${index < categoryIndex ? "completed" : "pending"}`} />
                )}
              </div>
            );
          })}
        </div>

        <div className="category-chip">{currentCategory}</div>
        <p className="category-note">{CATEGORY_INFO[currentCategory]}</p>

        <div className="category-questions">
          {currentQuestions.map((question) => (
            <div key={question.key} className="question-block">
              <p className="question-subtitle">{question.label}</p>
              {renderQuestionField(question)}
            </div>
          ))}
        </div>

        {error && <p className="error">{error}</p>}

        <div className="action-row">
          <button className="secondary-btn" onClick={handleBack} disabled={categoryIndex === 0 || loading}>
            Back
          </button>
          <button className="primary-btn" onClick={handleNext} disabled={loading}>
            {loading ? "Checking..." : isLastCategory ? "See Result" : "Next Category"}
          </button>
        </div>
      </section>
    </main>
  );
}

export default App;
