import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Leaf,
  Upload,
  Sparkles,
  Settings,
  X,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  RotateCcw,
  Copy,
  Check,
  History,
  Download,
  Code2,
  Server,
  FlaskConical,
  Image as ImageIcon,
  Info,
  Trash2,
  ChevronDown,
  ChevronUp,
  Loader2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from 'recharts';

const SAMPLE_LEAVES = [
  {
    id: 'sample-tomato-early',
    name: 'tomato_early_blight.jpg',
    label: 'Early Blight',
    diseaseKey: 'early_blight',
    sizeLabel: '148 KB',
    src: '/samples/tomato_early_blight.jpg',
    mockResult: {
      class: 'Tomato___Early_blight',
      confidence: 0.952,
      probabilities: {
        'Early Blight': 95.2,
        'Late Blight': 2.6,
        'Septoria Leaf Spot': 1.4,
        Healthy: 0.8,
      },
    },
  },
  {
    id: 'sample-tomato-late',
    name: 'tomato_late_blight.jpg',
    label: 'Late Blight',
    diseaseKey: 'late_blight',
    sizeLabel: '161 KB',
    src: '/samples/tomato_late_blight.jpg',
    mockResult: {
      class: 'Tomato___Late_blight',
      confidence: 0.969,
      probabilities: {
        'Late Blight': 96.9,
        'Early Blight': 1.9,
        'Leaf Mold': 0.8,
        Healthy: 0.4,
      },
    },
  },
  {
    id: 'sample-tomato-healthy',
    name: 'tomato_healthy_leaf.jpg',
    label: 'Healthy Leaf',
    diseaseKey: 'healthy',
    sizeLabel: '134 KB',
    src: '/samples/tomato_healthy_leaf.jpg',
    mockResult: {
      class: 'Tomato___healthy',
      confidence: 0.986,
      probabilities: {
        Healthy: 98.6,
        'Early Blight': 0.7,
        'Leaf Mold': 0.4,
        'Late Blight': 0.3,
      },
    },
  },
];

const DISEASE_PROFILES = {
  early_blight: {
    title: 'Tomato Early Blight',
    pathogen: 'Alternaria solani (Fungus)',
    severity: 'Moderate Risk',
    color: '#d97706',
    badgeClass: 'bg-amber-50 text-amber-800 border-amber-200',
    accentBg: 'bg-amber-50/70 border-amber-200/80',
    iconType: 'warning',
    summary:
      'Fungal disease producing dark brown concentric bullseye spots on older lower leaves, typically edged with a yellow halo.',
    actions: [
      'Prune affected lower tomato leaves and dispose of them away from the garden.',
      'Apply a copper-based, chlorothalonil, or mancozeb fungicide following label instructions.',
      'Water at the soil base in the morning and mulch around stems to prevent soil splash.',
      'Stake or cage tomato vines for better airflow and rotate nightshade crops every 2–3 years.',
    ],
  },
  late_blight: {
    title: 'Tomato Late Blight',
    pathogen: 'Phytophthora infestans (Oomycete)',
    severity: 'High Risk',
    color: '#dc2626',
    badgeClass: 'bg-rose-50 text-rose-800 border-rose-200',
    accentBg: 'bg-rose-50/70 border-rose-200/80',
    iconType: 'danger',
    summary:
      'Fast-spreading oomycete infection causing large dark water-soaked blotches on tomato leaflets and stems, with white fuzzy growth in humid weather.',
    actions: [
      'Remove and bag severely infected leaves or plants immediately—do not compost them.',
      'Apply targeted late-blight fungicides promptly when cool, damp weather is forecast.',
      'Avoid overhead watering and improve greenhouse or row ventilation to keep foliage dry.',
      'Use resistant tomato varieties and certified disease-free transplants next season.',
    ],
  },
  bacterial_spot: {
    title: 'Tomato Bacterial Spot',
    pathogen: 'Xanthomonas spp. (Bacteria)',
    severity: 'Moderate Risk',
    color: '#ea580c',
    badgeClass: 'bg-orange-50 text-orange-800 border-orange-200',
    accentBg: 'bg-orange-50/70 border-orange-200/80',
    iconType: 'warning',
    summary:
      'Bacterial infection causing small, dark water-soaked leaf spots that turn greasy brown-black and may cause leaf drop.',
    actions: [
      'Avoid working with wet tomato plants to prevent spreading bacteria by hand or tools.',
      'Apply fixed copper bactericide sprays early to slow further spread.',
      'Sanitize pruners and stakes with diluted bleach or alcohol between plants.',
    ],
  },
  leaf_mold: {
    title: 'Tomato Leaf Mold',
    pathogen: 'Passalora fulva (Fungus)',
    severity: 'Moderate Risk',
    color: '#ca8a04',
    badgeClass: 'bg-yellow-50 text-yellow-800 border-yellow-200',
    accentBg: 'bg-yellow-50/70 border-yellow-200/80',
    iconType: 'warning',
    summary:
      'Common in humid greenhouses; causes pale yellow patches on upper leaf surfaces and olive-green velvety mold underneath.',
    actions: [
      'Lower humidity below 85% with fans, venting, and wider plant spacing.',
      'Remove infected lower foliage and avoid wetting leaves during irrigation.',
      'Use a labeled fungicide if infection continues spreading upward.',
    ],
  },
  septoria_leaf_spot: {
    title: 'Septoria Leaf Spot',
    pathogen: 'Septoria lycopersici (Fungus)',
    severity: 'Moderate Risk',
    color: '#d97706',
    badgeClass: 'bg-amber-50 text-amber-800 border-amber-200',
    accentBg: 'bg-amber-50/70 border-amber-200/80',
    iconType: 'warning',
    summary:
      'Numerous small circular spots with dark brown borders and tan or gray centers appearing first on lower tomato leaves.',
    actions: [
      'Strip infected lower leaves and mulch soil to block rain-splashed spores.',
      'Apply a protectant fungicide (chlorothalonil or copper) every 7–10 days during wet spells.',
      'Clear all tomato debris at season end so spores do not overwinter.',
    ],
  },
  spider_mites: {
    title: 'Two-Spotted Spider Mites',
    pathogen: 'Tetranychus urticae (Pest)',
    severity: 'Moderate Risk',
    color: '#b45309',
    badgeClass: 'bg-amber-50 text-amber-800 border-amber-200',
    accentBg: 'bg-amber-50/70 border-amber-200/80',
    iconType: 'warning',
    summary:
      'Tiny sap-feeding mites causing yellow stippling, bronzing, and fine webbing on the undersides of tomato leaves.',
    actions: [
      'Spray undersides of leaves with a strong stream of water to dislodge mites.',
      'Apply insecticidal soap or horticultural neem oil thoroughly beneath leaves.',
      'Avoid broad-spectrum insecticides that wipe out predatory mites.',
    ],
  },
  target_spot: {
    title: 'Tomato Target Spot',
    pathogen: 'Corynespora cassiicola (Fungus)',
    severity: 'Moderate Risk',
    color: '#d97706',
    badgeClass: 'bg-amber-50 text-amber-800 border-amber-200',
    accentBg: 'bg-amber-50/70 border-amber-200/80',
    iconType: 'warning',
    summary:
      'Small brown lesions that expand into concentric rings on tomato leaves and fruit in warm, humid climates.',
    actions: [
      'Prune dense inner foliage to improve airflow and dry leaves quickly.',
      'Apply a broad-spectrum protectant fungicide on a regular schedule.',
      'Remove crop residues and rotate away from solanaceous crops.',
    ],
  },
  yellow_leaf_curl: {
    title: 'Yellow Leaf Curl Virus',
    pathogen: 'Tomato yellow leaf curl virus (TYLCV)',
    severity: 'High Risk',
    color: '#dc2626',
    badgeClass: 'bg-rose-50 text-rose-800 border-rose-200',
    accentBg: 'bg-rose-50/70 border-rose-200/80',
    iconType: 'danger',
    summary:
      'Whitefly-transmitted virus causing upward cupping, yellowing margins, and stunted growth on young tomato leaves.',
    actions: [
      'Rogue out and bag infected plants early to stop whiteflies from spreading the virus.',
      'Control silverleaf whiteflies using yellow sticky traps, reflective mulch, or insecticidal soap.',
      'Plant TYLCV-resistant tomato hybrids for future crops.',
    ],
  },
  mosaic_virus: {
    title: 'Tomato Mosaic Virus',
    pathogen: 'Tomato mosaic virus (ToMV)',
    severity: 'High Risk',
    color: '#dc2626',
    badgeClass: 'bg-rose-50 text-rose-800 border-rose-200',
    accentBg: 'bg-rose-50/70 border-rose-200/80',
    iconType: 'danger',
    summary:
      'Contact-transmitted virus causing light-and-dark green mottling, distortion, and fern-like narrowing of tomato leaflets.',
    actions: [
      'Remove infected plants carefully and wash hands and tools with soap or 10% milk solution.',
      'Avoid tobacco use near tomato plants and disinfect pruning shears frequently.',
      'Choose ToMV-resistant tomato cultivars when replanting.',
    ],
  },
  healthy: {
    title: 'Healthy Tomato Leaf',
    pathogen: 'No pathogen detected',
    severity: 'Healthy',
    color: '#059669',
    badgeClass: 'bg-emerald-50 text-emerald-800 border-emerald-200',
    accentBg: 'bg-emerald-50/70 border-emerald-200/80',
    iconType: 'healthy',
    summary:
      'Tomato foliage looks vibrant, evenly colored, and free of blight lesions, mold patches, or viral mottling.',
    actions: [
      'Keep scouting lower leaves weekly, especially after rain or humid nights.',
      'Water consistently at the root zone to prevent foliar wetness and blossom-end rot.',
      'Maintain balanced fertilization and sturdy trellising for good airflow.',
    ],
  },
};

function formatBytes(bytes) {
  if (!bytes || Number.isNaN(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function classifyDiseaseKey(rawLabel) {
  const s = String(rawLabel || '').toLowerCase();
  if (s.includes('late_blight') || s.includes('late blight')) return 'late_blight';
  if (s.includes('early_blight') || s.includes('early blight')) return 'early_blight';
  if (s.includes('bacterial')) return 'bacterial_spot';
  if (s.includes('leaf_mold') || s.includes('leaf mold')) return 'leaf_mold';
  if (s.includes('septoria')) return 'septoria_leaf_spot';
  if (s.includes('spider') || s.includes('mite')) return 'spider_mites';
  if (s.includes('target')) return 'target_spot';
  if (s.includes('curl') || s.includes('tylcv')) return 'yellow_leaf_curl';
  if (s.includes('mosaic') || s.includes('tomv')) return 'mosaic_virus';
  if (s.includes('healthy')) return 'healthy';
  return 'early_blight';
}

function cleanClassLabel(rawLabel) {
  const key = classifyDiseaseKey(rawLabel);
  if (DISEASE_PROFILES[key] && String(rawLabel).toLowerCase().includes('tomato')) {
    return DISEASE_PROFILES[key].title.replace(/^Tomato\s+/, '');
  }
  return String(rawLabel || 'Unknown')
    .replace(/^Tomato___/i, '')
    .replace(/___/g, ' - ')
    .replace(/_/g, ' ')
    .trim();
}

function normalizeConfidence(val) {
  const num = Number(val);
  if (Number.isNaN(num)) return 0;
  return num <= 1 ? Number((num * 100).toFixed(1)) : Number(num.toFixed(1));
}

function normalizeBackendResponse(rawJson) {
  const rawLabel =
    rawJson.class ??
    rawJson.prediction ??
    rawJson.disease ??
    rawJson.label ??
    rawJson.predicted_class ??
    'Unknown';

  const rawConf =
    rawJson.confidence ??
    rawJson.probability ??
    rawJson.score ??
    rawJson.accuracy ??
    0;

  const diseaseKey = classifyDiseaseKey(rawLabel);
  const confidencePct = Math.min(100, Math.max(0, normalizeConfidence(rawConf)));

  let chartData = [];
  if (Array.isArray(rawJson.all_predictions) && rawJson.all_predictions.length > 0) {
    chartData = rawJson.all_predictions.slice(0, 5).map((item) => {
      const lbl = item.display_name || item.class_label || 'Class';
      const key = classifyDiseaseKey(item.class_label || lbl);
      const pct = normalizeConfidence(item.percentage ?? item.probability ?? 0);
      return {
        name: cleanClassLabel(lbl),
        value: pct,
        color: DISEASE_PROFILES[key]?.color || '#dc2626',
      };
    });
  } else {
    const probs = rawJson.probabilities || rawJson.scores;
    if (probs && typeof probs === 'object' && !Array.isArray(probs)) {
      chartData = Object.entries(probs)
        .map(([k, v]) => {
          const key = classifyDiseaseKey(k);
          return {
            name: cleanClassLabel(k),
            value: normalizeConfidence(v),
            color: DISEASE_PROFILES[key]?.color || '#dc2626',
          };
        })
        .sort((a, b) => b.value - a.value)
        .slice(0, 5);
    } else {
      const remaining = Math.max(0, Number((100 - confidencePct).toFixed(1)));
      const splitA = Number((remaining * 0.65).toFixed(1));
      const splitB = Number((remaining - splitA).toFixed(1));
      chartData = [
        {
          name: 'Early Blight',
          value: diseaseKey === 'early_blight' ? confidencePct : splitA,
          color: '#d97706',
        },
        {
          name: 'Late Blight',
          value: diseaseKey === 'late_blight' ? confidencePct : diseaseKey === 'early_blight' ? splitA : splitB,
          color: '#dc2626',
        },
        {
          name: 'Healthy',
          value: diseaseKey === 'healthy' ? confidencePct : splitB,
          color: '#059669',
        },
      ].sort((a, b) => b.value - a.value);
    }
  }

  return {
    diseaseKey,
    rawLabel: String(rawLabel),
    confidence: confidencePct,
    chartData,
    rawJson,
  };
}

function analyzeCustomImageDemo(imageUrl) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const w = 64;
        const h = 64;
        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(img, 0, 0, w, h);
        const data = ctx.getImageData(0, 0, w, h).data;
        let darkSpots = 0;
        let yellowBrown = 0;
        let greenPixels = 0;
        const total = w * h;
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          if (g > r * 1.08 && g > b * 1.08 && g > 50) {
            greenPixels++;
          } else if (r < 65 && g < 65 && b < 60) {
            darkSpots++;
          } else if (r > 95 && g > 75 && b < 75) {
            yellowBrown++;
          }
        }
        const darkRatio = darkSpots / total;
        const brownRatio = yellowBrown / total;
        const greenRatio = greenPixels / total;

        let payload;
        if (darkRatio > 0.18) {
          const conf = Math.min(97.2, Number((86 + darkRatio * 28).toFixed(1)));
          const rem = Number((100 - conf).toFixed(1));
          payload = {
            class: 'Tomato___Late_blight',
            confidence: conf / 100,
            probabilities: {
              'Late Blight': conf,
              'Early Blight': Number((rem * 0.62).toFixed(1)),
              'Leaf Mold': Number((rem * 0.24).toFixed(1)),
              Healthy: Number((rem * 0.14).toFixed(1)),
            },
          };
        } else if (brownRatio > 0.14 || greenRatio < 0.35) {
          const conf = Math.min(95.6, Number((84 + brownRatio * 30).toFixed(1)));
          const rem = Number((100 - conf).toFixed(1));
          payload = {
            class: 'Tomato___Early_blight',
            confidence: conf / 100,
            probabilities: {
              'Early Blight': conf,
              'Septoria Leaf Spot': Number((rem * 0.52).toFixed(1)),
              'Late Blight': Number((rem * 0.32).toFixed(1)),
              Healthy: Number((rem * 0.16).toFixed(1)),
            },
          };
        } else {
          const conf = Math.min(98.4, Number((88 + greenRatio * 14).toFixed(1)));
          const rem = Number((100 - conf).toFixed(1));
          payload = {
            class: 'Tomato___healthy',
            confidence: conf / 100,
            probabilities: {
              Healthy: conf,
              'Early Blight': Number((rem * 0.5).toFixed(1)),
              'Leaf Mold': Number((rem * 0.3).toFixed(1)),
              'Late Blight': Number((rem * 0.2).toFixed(1)),
            },
          };
        }
        resolve(normalizeBackendResponse(payload));
      } catch {
        resolve(normalizeBackendResponse(SAMPLE_LEAVES[0].mockResult));
      }
    };
    img.onerror = () => {
      resolve(normalizeBackendResponse(SAMPLE_LEAVES[0].mockResult));
    };
    img.src = imageUrl;
  });
}

function copyTextFallback(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text).catch(() => {
      const el = document.createElement('textarea');
      el.value = text;
      el.style.position = 'fixed';
      el.style.opacity = '0';
      document.body.appendChild(el);
      el.select();
      document.execCommand('copy');
      document.body.removeChild(el);
    });
  }
  const el = document.createElement('textarea');
  el.value = text;
  el.style.position = 'fixed';
  el.style.opacity = '0';
  document.body.appendChild(el);
  el.select();
  document.execCommand('copy');
  document.body.removeChild(el);
  return Promise.resolve();
}

export default function App() {
  const [mode, setMode] = useState(import.meta.env.VITE_API_URL ? 'live' : 'demo');
  const [apiUrl, setApiUrl] = useState(import.meta.env.VITE_API_URL || 'http://localhost:8000/predict');
  const [fieldName, setFieldName] = useState(import.meta.env.VITE_API_FIELD || 'file');
  const [authHeader, setAuthHeader] = useState('');
  const [showSettings, setShowSettings] = useState(false);

  const [selectedImage, setSelectedImage] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [showRawJson, setShowRawJson] = useState(false);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState([]);
  const [toast, setToast] = useState('');

  const fileInputRef = useRef(null);

  useEffect(() => {
    try {
      const savedUrl = localStorage.getItem('tomato_doctor_api_url');
      const savedField = localStorage.getItem('tomato_doctor_field_name');
      if (savedUrl) setApiUrl(savedUrl);
      if (savedField) setFieldName(savedField);
    } catch {
      // Ignore storage access errors in restricted frames
    }
  }, []);

  const showToastMsg = useCallback((msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 2600);
  }, []);

  const handleSaveSettings = (e) => {
    e.preventDefault();
    try {
      localStorage.setItem('tomato_doctor_api_url', apiUrl.trim());
      localStorage.setItem('tomato_doctor_field_name', fieldName.trim() || 'file');
    } catch {
      // Ignore storage errors
    }
    setShowSettings(false);
    showToastMsg('Backend settings saved');
  };

  const handleFilePicked = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please upload a valid image file (JPG, PNG, or WEBP).');
      return;
    }
    setErrorMsg('');
    setResult(null);
    const reader = new FileReader();
    reader.onload = (ev) => {
      setSelectedImage({
        src: ev.target.result,
        name: file.name,
        sizeLabel: formatBytes(file.size),
        file,
        sampleId: null,
      });
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleFilePicked(files[0]);
    }
  };

  const handleSelectSample = (sample) => {
    setErrorMsg('');
    setResult(null);
    setSelectedImage({
      src: sample.src,
      name: sample.name,
      sizeLabel: sample.sizeLabel,
      file: null,
      sampleId: sample.id,
    });
  };

  const handleClear = () => {
    setSelectedImage(null);
    setResult(null);
    setErrorMsg('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    setErrorMsg('');
    setIsAnalyzing(true);

    try {
      let normalized;

      if (mode === 'live') {
        const trimmedUrl = apiUrl.trim();
        if (!trimmedUrl) {
          setIsAnalyzing(false);
          setShowSettings(true);
          setErrorMsg('Enter your deployed HTTPS backend endpoint URL in API Settings first.');
          return;
        }
        if (!trimmedUrl.startsWith('http://') && !trimmedUrl.startsWith('https://')) {
          setIsAnalyzing(false);
          setErrorMsg('Enter a valid http:// or https:// backend URL in API Settings.');
          return;
        }

        const formData = new FormData();
        const formKey = fieldName.trim() || 'file';

        if (selectedImage.file) {
          formData.append(formKey, selectedImage.file);
        } else {
          const blobRes = await fetch(selectedImage.src);
          const blob = await blobRes.blob();
          formData.append(formKey, blob, selectedImage.name);
        }

        const headers = {};
        if (authHeader.trim()) {
          headers['Authorization'] = authHeader.trim();
        }

        const response = await fetch(trimmedUrl, {
          method: 'POST',
          headers,
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Backend responded with status ${response.status} (${response.statusText || 'Error'})`);
        }

        const data = await response.json();
        normalized = normalizeBackendResponse(data);
      } else {
        await new Promise((r) => setTimeout(r, 750));
        if (selectedImage.sampleId) {
          const sample = SAMPLE_LEAVES.find((s) => s.id === selectedImage.sampleId);
          normalized = normalizeBackendResponse(sample.mockResult);
        } else {
          normalized = await analyzeCustomImageDemo(selectedImage.src);
        }
      }

      setResult(normalized);

      const scanEntry = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        mode,
        image: selectedImage,
        result: normalized,
      };
      setHistory((prev) => [scanEntry, ...prev].slice(0, 8));
    } catch (err) {
      setErrorMsg(
        err?.message ||
          'Could not reach the backend API. Check your endpoint URL or switch to Demo Mode.'
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCopyJson = () => {
    if (!result) return;
    copyTextFallback(JSON.stringify(result.rawJson, null, 2)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };

  const handleExportHistory = () => {
    if (history.length === 0) return;
    const exportPayload = history.map((item) => ({
      timestamp: item.timestamp,
      filename: item.image.name,
      mode: item.mode,
      prediction: DISEASE_PROFILES[item.result.diseaseKey]?.title || item.result.rawLabel,
      confidence_percent: item.result.confidence,
      raw_response: item.result.rawJson,
    }));
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tomato-disease-scans.json';
    a.click();
    URL.revokeObjectURL(url);
    showToastMsg('Exported scan history as JSON');
  };

  const profile = result ? DISEASE_PROFILES[result.diseaseKey] : null;

  const renderStatusIcon = (iconType) => {
    if (iconType === 'healthy') return <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />;
    if (iconType === 'danger') return <ShieldAlert className="w-5 h-5 text-rose-600 shrink-0" />;
    return <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0" />;
  };

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900 font-sans">
      {/* Top Navigation */}
      <header className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b border-stone-200/80">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-9 h-9 rounded-xl bg-red-600 text-white flex items-center justify-center shadow-sm shrink-0">
              <Leaf className="w-5 h-5" />
            </div>
            <div className="truncate">
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-stone-900 leading-tight truncate">
                Tomato Doctor AI
              </h1>
              <p className="text-xs text-stone-500 hidden sm:block">
                Tomato Leaf Disease Detector
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            <div className="inline-flex items-center rounded-lg bg-stone-100 p-1 border border-stone-200/80">
              <button
                type="button"
                onClick={() => {
                  setMode('demo');
                  setErrorMsg('');
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  mode === 'demo'
                    ? 'bg-white text-red-700 shadow-sm'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                <FlaskConical className="w-3.5 h-3.5" />
                <span>Demo</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setMode('live');
                  setErrorMsg('');
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  mode === 'live'
                    ? 'bg-white text-red-700 shadow-sm'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                <Server className="w-3.5 h-3.5" />
                <span>Live API</span>
              </button>
            </div>

            <button
              type="button"
              onClick={() => setShowSettings(true)}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-stone-200 bg-white hover:bg-stone-50 text-xs font-semibold text-stone-700 transition-colors"
            >
              <Settings className="w-3.5 h-3.5 text-stone-500" />
              <span className="hidden sm:inline">API Settings</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-8 space-y-6">
        {mode === 'live' && (
          <div className="rounded-xl border border-red-200 bg-red-50/70 px-4 py-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 text-xs sm:text-sm text-red-900">
              <Server className="w-4 h-4 text-red-700 shrink-0" />
              <span>
                <strong>Live Backend Mode:</strong>{' '}
                {apiUrl ? (
                  <code className="px-1.5 py-0.5 rounded bg-white/80 text-red-900 font-mono text-xs border border-red-200">
                    POST {apiUrl}
                  </code>
                ) : (
                  'No HTTPS endpoint configured yet.'
                )}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setShowSettings(true)}
              className="text-xs font-semibold text-red-800 hover:text-red-950 underline underline-offset-2"
            >
              Configure Endpoint
            </button>
          </div>
        )}

        {/* Primary 2-Column Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Drag & Drop Uploader */}
          <section className="lg:col-span-6 bg-white rounded-2xl border border-stone-200/90 shadow-sm p-5 sm:p-6 space-y-5">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h2 className="text-base font-bold text-stone-900">Upload Tomato Leaf</h2>
                <p className="text-xs text-stone-500 mt-0.5">
                  Drag and drop a clear tomato leaf photo or browse from your device
                </p>
              </div>
              {selectedImage && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium text-stone-600 hover:text-rose-700 hover:bg-rose-50 transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Clear
                </button>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={(e) => handleFilePicked(e.target.files?.[0])}
            />

            {!selectedImage ? (
              <div
                onDragEnter={handleDragOver}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
                }}
                className={`relative rounded-2xl border-2 border-dashed p-8 sm:p-10 text-center cursor-pointer transition-all ${
                  isDragging
                    ? 'border-red-500 bg-red-50/70 scale-[1.01]'
                    : 'border-stone-300 bg-stone-50/70 hover:border-red-400 hover:bg-red-50/30'
                }`}
              >
                <div className="mx-auto w-14 h-14 rounded-2xl bg-red-100 text-red-600 flex items-center justify-center mb-4">
                  <Upload className="w-7 h-7" />
                </div>
                <p className="text-sm font-semibold text-stone-800">
                  {isDragging ? 'Drop your tomato leaf image here' : 'Drag & drop a tomato leaf photo'}
                </p>
                <p className="text-xs text-stone-500 mt-1.5">
                  or <span className="text-red-600 font-semibold underline underline-offset-2">click to browse files</span>
                </p>
                <p className="text-[11px] text-stone-400 mt-3">
                  Supports JPG, PNG, WEBP • Works with Early/Late Blight, Leaf Mold, Septoria, Viruses & more
                </p>
              </div>
            ) : (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`rounded-2xl border overflow-hidden transition-all ${
                  isDragging ? 'border-red-500 ring-2 ring-red-200' : 'border-stone-200'
                }`}
              >
                <div className="relative bg-stone-900/5 aspect-[4/3] w-full overflow-hidden">
                  <img
                    src={selectedImage.src}
                    alt={selectedImage.name}
                    className="w-full h-full object-cover"
                  />
                  {isAnalyzing && (
                    <div className="absolute inset-0 bg-stone-950/45 backdrop-blur-[2px] flex flex-col items-center justify-center text-white gap-2">
                      <Loader2 className="w-8 h-8 animate-spin text-red-300" />
                      <span className="text-xs font-semibold tracking-wide uppercase">
                        Analyzing Tomato Leaf...
                      </span>
                    </div>
                  )}
                </div>
                <div className="px-4 py-3 bg-stone-50 border-t border-stone-200/80 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <ImageIcon className="w-4 h-4 text-red-600 shrink-0" />
                    <span className="text-xs font-semibold text-stone-800 truncate">
                      {selectedImage.name}
                    </span>
                    <span className="text-[11px] text-stone-400 shrink-0">
                      ({selectedImage.sizeLabel})
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-xs font-semibold text-red-600 hover:text-red-800 shrink-0"
                  >
                    Replace
                  </button>
                </div>
              </div>
            )}

            {errorMsg && (
              <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-800 flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <div className="space-y-1.5">
                  <p className="font-medium leading-relaxed">{errorMsg}</p>
                  {mode === 'live' && (
                    <button
                      type="button"
                      onClick={() => {
                        setMode('demo');
                        setErrorMsg('');
                      }}
                      className="text-xs font-semibold text-rose-900 underline underline-offset-2"
                    >
                      Switch to Demo Mode
                    </button>
                  )}
                </div>
              </div>
            )}

            <button
              type="button"
              onClick={handleAnalyze}
              disabled={!selectedImage || isAnalyzing}
              className={`w-full py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all shadow-sm ${
                !selectedImage || isAnalyzing
                  ? 'bg-stone-200 text-stone-400 cursor-not-allowed'
                  : 'bg-red-600 hover:bg-red-700 active:scale-[0.99] text-white'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Running Disease Detection...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>
                    {mode === 'live' ? 'Send to Backend & Diagnose' : 'Analyze Tomato Leaf'}
                  </span>
                </>
              )}
            </button>

            <div className="pt-2 border-t border-stone-100">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-xs font-semibold text-stone-600">
                  Quick Test Samples
                </span>
                <span className="text-[11px] text-stone-400">Click any leaf to load</span>
              </div>
              <div className="grid grid-cols-3 gap-2.5">
                {SAMPLE_LEAVES.map((sample) => {
                  const isActive = selectedImage?.sampleId === sample.id;
                  return (
                    <button
                      key={sample.id}
                      type="button"
                      onClick={() => handleSelectSample(sample)}
                      className={`group rounded-xl border overflow-hidden text-left transition-all ${
                        isActive
                          ? 'border-red-600 ring-2 ring-red-100'
                          : 'border-stone-200 hover:border-red-400'
                      }`}
                    >
                      <div className="aspect-[4/3] w-full overflow-hidden bg-stone-100">
                        <img
                          src={sample.src}
                          alt={sample.label}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        />
                      </div>
                      <div className="px-2 py-1.5 bg-white">
                        <p className="text-[11px] font-semibold text-stone-800 truncate">
                          {sample.label}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </section>

          {/* Right Column: Diagnosis Results */}
          <section className="lg:col-span-6 bg-white rounded-2xl border border-stone-200/90 shadow-sm p-5 sm:p-6">
            {!result ? (
              <div className="py-10 sm:py-14 text-center space-y-5">
                <div className="mx-auto w-14 h-14 rounded-2xl bg-red-50 text-red-600 flex items-center justify-center">
                  <Leaf className="w-7 h-7" />
                </div>
                <div className="max-w-sm mx-auto space-y-1.5">
                  <h2 className="text-base font-bold text-stone-900">Ready for Tomato Diagnosis</h2>
                  <p className="text-xs text-stone-500 leading-relaxed">
                    Drop a tomato leaf photo on the left or pick a sample leaf, then click{' '}
                    <span className="font-semibold text-stone-700">Analyze Tomato Leaf</span> to view
                    predicted disease class, confidence scores, and treatment guidance.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-left pt-4 max-w-lg mx-auto">
                  <div className="rounded-xl bg-stone-50 border border-stone-200/70 p-3">
                    <span className="text-[11px] font-bold text-red-600 uppercase">Step 1</span>
                    <p className="text-xs font-semibold text-stone-800 mt-1">Drop Leaf Photo</p>
                    <p className="text-[11px] text-stone-500 mt-0.5">JPG, PNG, or WEBP</p>
                  </div>
                  <div className="rounded-xl bg-stone-50 border border-stone-200/70 p-3">
                    <span className="text-[11px] font-bold text-red-600 uppercase">Step 2</span>
                    <p className="text-xs font-semibold text-stone-800 mt-1">AI Classification</p>
                    <p className="text-[11px] text-stone-500 mt-0.5">10 Tomato Classes</p>
                  </div>
                  <div className="rounded-xl bg-stone-50 border border-stone-200/70 p-3">
                    <span className="text-[11px] font-bold text-red-600 uppercase">Step 3</span>
                    <p className="text-xs font-semibold text-stone-800 mt-1">Action Plan</p>
                    <p className="text-[11px] text-stone-500 mt-0.5">Treatment & JSON log</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                <div className={`rounded-xl border p-4 ${profile.accentBg}`}>
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-2.5">
                      {renderStatusIcon(profile.iconType)}
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <h2 className="text-lg font-bold text-stone-900 leading-tight">
                            {profile.title}
                          </h2>
                          <span
                            className={`px-2 py-0.5 rounded-full text-[11px] font-semibold border ${profile.badgeClass}`}
                          >
                            {profile.severity}
                          </span>
                        </div>
                        <p className="text-xs text-stone-600 italic mt-0.5">{profile.pathogen}</p>
                      </div>
                    </div>

                    <div className="text-right shrink-0">
                      <span className="text-2xl font-extrabold text-stone-900">
                        {result.confidence}%
                      </span>
                      <p className="text-[11px] text-stone-500 font-medium">Confidence</p>
                    </div>
                  </div>

                  <div className="mt-3.5 h-2 w-full rounded-full bg-white/80 overflow-hidden border border-stone-200/60">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${result.confidence}%`,
                        backgroundColor: profile.color,
                      }}
                    />
                  </div>
                  <p className="text-xs text-stone-700 mt-3 leading-relaxed">{profile.summary}</p>
                </div>

                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-stone-500 mb-2">
                    Class Probability Breakdown
                  </h3>
                  <div className="relative h-48 w-full rounded-xl border border-stone-200/80 bg-stone-50/50 p-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={result.chartData}
                        layout="vertical"
                        margin={{ top: 6, right: 24, left: 16, bottom: 6 }}
                      >
                        <XAxis
                          type="number"
                          domain={[0, 100]}
                          tickFormatter={(v) => `${v}%`}
                          tick={{ fontSize: 11, fill: '#78716c' }}
                        />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={110}
                          tick={{ fontSize: 11, fill: '#292524', fontWeight: 600 }}
                        />
                        <Tooltip
                          formatter={(val) => [`${val}%`, 'Probability']}
                          contentStyle={{
                            borderRadius: '10px',
                            border: '1px solid #e7e5e4',
                            fontSize: '12px',
                          }}
                        />
                        <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={20}>
                          {result.chartData.map((entry, idx) => (
                            <Cell key={`cell-${idx}`} fill={entry.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-stone-500 mb-2.5">
                    Recommended Next Steps
                  </h3>
                  <ul className="space-y-2">
                    {profile.actions.map((step, idx) => (
                      <li
                        key={idx}
                        className="flex items-start gap-2.5 text-xs text-stone-700 leading-relaxed bg-stone-50 rounded-lg px-3 py-2 border border-stone-200/60"
                      >
                        <span className="w-4 h-4 rounded-full bg-red-100 text-red-700 text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="border-t border-stone-100 pt-3">
                  <div className="flex items-center justify-between">
                    <button
                      type="button"
                      onClick={() => setShowRawJson((v) => !v)}
                      className="inline-flex items-center gap-1.5 text-xs font-semibold text-stone-600 hover:text-stone-900"
                    >
                      <Code2 className="w-3.5 h-3.5" />
                      <span>API Response Payload</span>
                      {showRawJson ? (
                        <ChevronUp className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5" />
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={handleCopyJson}
                      className="inline-flex items-center gap-1 text-xs font-semibold text-red-600 hover:text-red-800"
                    >
                      {copied ? (
                        <>
                          <Check className="w-3.5 h-3.5" />
                          <span>Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy JSON</span>
                        </>
                      )}
                    </button>
                  </div>

                  {showRawJson && (
                    <pre className="mt-2.5 p-3 rounded-xl bg-stone-900 text-emerald-300 text-[11px] font-mono overflow-x-auto leading-relaxed">
                      {JSON.stringify(result.rawJson, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>

        {/* Recent Scans History */}
        {history.length > 0 && (
          <section className="bg-white rounded-2xl border border-stone-200/90 shadow-sm p-5 sm:p-6">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-red-600" />
                <h3 className="text-sm font-bold text-stone-900">
                  Recent Scans ({history.length})
                </h3>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleExportHistory}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-stone-200 text-xs font-semibold text-stone-700 hover:bg-stone-50"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export JSON
                </button>
                <button
                  type="button"
                  onClick={() => setHistory([])}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-stone-500 hover:text-rose-700 hover:bg-rose-50"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {history.map((item) => {
                const itemProfile = DISEASE_PROFILES[item.result.diseaseKey];
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setSelectedImage(item.image);
                      setResult(item.result);
                    }}
                    className="flex items-center gap-3 p-2.5 rounded-xl border border-stone-200 hover:border-red-400 hover:bg-stone-50 text-left transition-all"
                  >
                    <img
                      src={item.image.src}
                      alt={item.image.name}
                      className="w-12 h-12 rounded-lg object-cover shrink-0 bg-stone-100"
                    />
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-bold text-stone-900 truncate">
                        {itemProfile.title}
                      </p>
                      <p className="text-[11px] text-stone-500 truncate">{item.image.name}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-[11px] font-semibold text-red-600">
                          {item.result.confidence}%
                        </span>
                        <span className="text-[10px] text-stone-400">{item.timestamp}</span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </section>
        )}
      </main>

      {/* API Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 bg-stone-950/50 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl border border-stone-200 shadow-xl max-w-md w-full overflow-hidden">
            <div className="px-5 py-4 border-b border-stone-100 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Server className="w-4 h-4 text-red-600" />
                <h3 className="text-sm font-bold text-stone-900">Backend API Configuration</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowSettings(false)}
                className="p-1 rounded-lg text-stone-400 hover:text-stone-700 hover:bg-stone-100"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleSaveSettings} className="p-5 space-y-4">
              <div>
                <label className="block text-xs font-semibold text-stone-700 mb-1">
                  Prediction Endpoint URL
                </label>
                <input
                  type="url"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  placeholder="http://localhost:8000/predict"
                  className="w-full px-3 py-2 rounded-lg border border-stone-300 text-xs focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-red-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-stone-700 mb-1">
                  Multipart Form Field Name
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={fieldName}
                    onChange={(e) => setFieldName(e.target.value)}
                    placeholder="file"
                    className="flex-1 px-3 py-2 rounded-lg border border-stone-300 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-red-500"
                  />
                  <button
                    type="button"
                    onClick={() => setFieldName('file')}
                    className="px-2.5 py-2 rounded-lg border border-stone-200 text-xs font-mono hover:bg-stone-50"
                  >
                    file
                  </button>
                  <button
                    type="button"
                    onClick={() => setFieldName('image')}
                    className="px-2.5 py-2 rounded-lg border border-stone-200 text-xs font-mono hover:bg-stone-50"
                  >
                    image
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-stone-700 mb-1">
                  Authorization Header (Optional)
                </label>
                <input
                  type="text"
                  value={authHeader}
                  onChange={(e) => setAuthHeader(e.target.value)}
                  placeholder="Bearer your-token-if-needed"
                  className="w-full px-3 py-2 rounded-lg border border-stone-300 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-red-500"
                />
              </div>

              <div className="rounded-xl bg-stone-50 border border-stone-200/80 p-3 flex items-start gap-2 text-[11px] text-stone-600 leading-relaxed">
                <Info className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                <span>
                  Sends <code className="font-mono text-stone-800">POST multipart/form-data</code>{' '}
                  and parses PlantVillage tomato labels such as{' '}
                  <code className="font-mono text-stone-800">Tomato___Early_blight</code>,{' '}
                  <code className="font-mono text-stone-800">Tomato___Late_blight</code>,{' '}
                  <code className="font-mono text-stone-800">Tomato___Leaf_Mold</code>, and more.
                </span>
              </div>

              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowSettings(false)}
                  className="px-3.5 py-2 rounded-lg text-xs font-semibold text-stone-600 hover:bg-stone-100"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-semibold shadow-sm"
                >
                  Save Settings
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Toast Notification */}
      {toast && (
        <div className="fixed bottom-5 right-5 z-50 px-4 py-2.5 rounded-xl bg-stone-900 text-white text-xs font-semibold shadow-lg flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{toast}</span>
        </div>
      )}
    </div>
  );
}
