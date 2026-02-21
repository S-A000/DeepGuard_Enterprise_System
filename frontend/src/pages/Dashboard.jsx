import React, { useState, useEffect } from "react";
import axios from 'axios';
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";

const Dashboard = () => {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [userName, setUserName] = useState("Investigator");

    useEffect(() => {
        const savedName = localStorage.getItem('userName');
        if (savedName) setUserName(savedName);
    }, []);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setResult(null); 
    };

    const handleUpload = async () => {
        if (!file) {
            alert("Bhai, pehle koi video select toh karo!");
            return;
        }

        const userID = localStorage.getItem('user_id') || 1;
        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', userID);

        setLoading(true);
        try {
            const response = await axios.post('http://127.0.0.1:8000/api/analyze', formData);
            setResult(response.data);
        } catch (error) {
            alert("Analysis failed. Backend is not responding.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={styles.pageWrapper}>
            {/* 1. Left Sidebar */}
            <Sidebar />

            {/* 2. Right Main Content Area */}
            <div style={styles.mainArea}>
                {/* Top Navbar */}
                <Navbar />

                <div style={styles.contentPadding}>
                    <header style={styles.welcomeHeader}>
                        <div>
                            <h2 style={{ margin: 0, color: '#f8fafc' }}>Forensic Workstation</h2>
                            <p style={{ color: '#94a3b8', fontSize: '14px' }}>Welcome back, {userName}</p>
                        </div>
                        <div style={styles.pulseBadge}>● AI Engine Active</div>
                    </header>

                    {/* --- Analysis Section --- */}
                    <div style={styles.uploadSection}>
                        <div style={styles.dropZone}>
                            <div style={{ fontSize: '40px', marginBottom: '10px' }}>📁</div>
                            <input type="file" onChange={handleFileChange} style={{ color: '#94a3b8' }} />
                            <p style={{ color: '#64748b', fontSize: '12px', marginTop: '10px' }}>
                                Upload evidence for Deepfake detection (MP4/AVI/MOV)
                            </p>
                        </div>

                        <button 
                            onClick={handleUpload} 
                            disabled={loading}
                            style={styles.scanBtn(loading)}
                            className={loading ? "pulse" : ""}
                        >
                            {loading ? "🧬 ANALYZING BIOMETRIC MARKERS..." : "🚀 START FORENSIC SCAN"}
                        </button>
                    </div>

                    {/* --- Detailed Results Area --- */}
                    {result && (
                        <div style={{ marginTop: '30px', animation: 'slideUp 0.6s ease' }}>
                            <div style={styles.verdictBanner(result.verdict)}>
                                <h1 style={{ margin: 0, fontSize: '3.2rem' }}>{result.verdict}</h1>
                                <p style={{ fontSize: '1.1rem', opacity: 0.9 }}>
                                    Confidence: <b>{result.confidence}%</b> | Processing: <b>{result.processing_time}</b>
                                </p>
                            </div>

                            <div style={styles.metricsGrid}>
                                <h3 style={styles.gridTitle}>🔬 TECHNICAL MEASUREMENTS</h3>
                                <div style={styles.gridContent}>
                                    <MetricBar label="Spatial Consistency" score={result.branch_scores?.spatial || 0} />
                                    <MetricBar label="Physical Integrity" score={result.branch_scores?.physics || 0} />
                                    <MetricBar label="Digital Forensics" score={result.branch_scores?.forensics || 0} />
                                    <MetricBar label="Audio Pattern" score={result.branch_scores?.audio || 0} />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            
            <style>{`
                @keyframes slideUp { from { transform: translateY(30px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
                .pulse { animation: pulseAnim 1.5s infinite; }
                @keyframes pulseAnim { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
            `}</style>
        </div>
    );
};

// Internal Progress Bar Component
const MetricBar = ({ label, score }) => (
    <div style={{ marginBottom: '15px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>{label}</span>
            <span style={{ fontSize: '13px', color: '#f8fafc', fontWeight: 'bold' }}>{score}%</span>
        </div>
        <div style={{ height: '6px', backgroundColor: '#1e293b', borderRadius: '3px' }}>
            <div style={{ 
                height: '100%', width: `${score}%`, 
                backgroundColor: score > 75 ? '#ef4444' : '#38bdf8', 
                borderRadius: '3px', transition: 'width 1.2s ease-in-out',
                boxShadow: `0 0 10px ${score > 75 ? '#ef444466' : '#38bdf866'}`
            }}></div>
        </div>
    </div>
);

const styles = {
    pageWrapper: { display: 'flex', height: '100vh', backgroundColor: '#020617', overflow: 'hidden' },
    mainArea: { flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto' },
    contentPadding: { padding: '40px' },
    welcomeHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' },
    pulseBadge: { padding: '6px 14px', backgroundColor: '#064e3b', color: '#10b981', borderRadius: '20px', fontSize: '12px', fontWeight: 'bold', border: '1px solid #10b981' },
    
    uploadSection: { backgroundColor: '#1e293b', padding: '30px', borderRadius: '20px', border: '1px solid #334155', boxShadow: '0 10px 30px rgba(0,0,0,0.3)' },
    dropZone: { border: '2px dashed #475569', padding: '40px', textAlign: 'center', borderRadius: '15px', backgroundColor: '#0f172a', marginBottom: '25px' },
    
    scanBtn: (loading) => ({
        width: '100%', padding: '16px', borderRadius: '12px', border: 'none', fontSize: '16px', fontWeight: 'bold',
        backgroundColor: loading ? '#334155' : '#38bdf8', color: '#0f172a', cursor: loading ? 'not-allowed' : 'pointer',
        transition: '0.3s ease'
    }),

    verdictBanner: (verdict) => ({
        padding: '35px', borderRadius: '20px', textAlign: 'center', color: 'white',
        background: verdict === "FAKE" ? 'linear-gradient(135deg, #991b1b, #450a0a)' : 'linear-gradient(135deg, #065f46, #064e3b)',
        border: `2px solid ${verdict === "FAKE" ? "#ef4444" : "#10b981"}`,
        boxShadow: `0 15px 40px ${verdict === "FAKE" ? "#ef444433" : "#10b98133"}`
    }),

    metricsGrid: { marginTop: '25px', padding: '30px', backgroundColor: '#0f172a', borderRadius: '20px', border: '1px solid #1e293b' },
    gridTitle: { color: '#38bdf8', fontSize: '16px', marginTop: 0, marginBottom: '25px', textAlign: 'center', letterSpacing: '1px' },
    gridContent: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }
};

export default Dashboard;