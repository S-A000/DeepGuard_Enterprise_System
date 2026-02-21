import React, { useState, useEffect } from "react";
import axios from "axios";
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";

const Dashboard = () => {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [userName, setUserName] = useState("Investigator");

    useEffect(() => {
        const savedName = localStorage.getItem("userName");
        if (savedName) setUserName(savedName);
    }, []);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreviewUrl(URL.createObjectURL(selectedFile));
            setResult(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return alert("⚠️ Select evidence first!");

        const userID = localStorage.getItem("user_id") || 1;
        const formData = new FormData();
        formData.append("file", file);
        formData.append("user_id", userID);

        setLoading(true);
        try {
            const response = await axios.post(
                "http://127.0.0.1:8000/api/analyze",
                formData
            );
            setResult(response.data);
        } catch (error) {
            alert("❌ System Error: Forensic Engine Offline.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={styles.pageWrapper}>
            <Sidebar />

            <div style={styles.mainArea}>
                <Navbar />

                <div style={styles.contentPadding}>
                    <header style={styles.welcomeHeader}>
                        <div>
                            <h1 style={styles.mainTitle}>Forensic Intelligence Console</h1>
                            <p style={styles.subTitle}>
                                Active Subject:{" "}
                                <span style={{ color: "#38bdf8" }}>{userName}</span>
                            </p>
                        </div>
                        <div style={styles.liveBadge}>● ENGINE ONLINE</div>
                    </header>

                    {/* Upload Section */}
                    <div style={styles.uploadTerminal}>
                        {!previewUrl ? (
                            <div style={styles.dropZone}>
                                <div style={{ fontSize: "55px" }}>📡</div>
                                <h3 style={{ color: "#f8fafc", marginTop: "10px" }}>
                                    Inject Evidence Stream
                                </h3>
                                <label style={styles.fileButton}>
                                    SELECT SOURCE
                                    <input
                                        type="file"
                                        onChange={handleFileChange}
                                        style={{ display: "none" }}
                                    />
                                </label>
                            </div>
                        ) : (
                            <div style={styles.previewContainer}>
                                <div style={styles.videoHeader}>
                                    EVIDENCE_PLAYBACK_MODULE
                                </div>
                                <video
                                    src={previewUrl}
                                    controls
                                    style={styles.videoElement}
                                />
                                <button
                                    onClick={() => {
                                        setFile(null);
                                        setPreviewUrl(null);
                                    }}
                                    style={styles.changeBtn}
                                >
                                    CHANGE SOURCE
                                </button>
                            </div>
                        )}

                        <button
                            onClick={handleUpload}
                            disabled={loading || !file}
                            style={styles.scanBtn(loading || !file)}
                            className={loading ? "pulse-glow" : ""}
                        >
                            {loading
                                ? "EXTRACTING SIGNAL MATRICES..."
                                : "INITIATE DEEP SCAN"}
                        </button>
                    </div>

                    {/* Results */}
                    {result && (
                        <div style={{ marginTop: "50px", animation: "slideUp 0.6s ease" }}>
                            <div style={styles.verdictPanel(result.verdict)}>
                                <h1 style={styles.verdictText}>
                                    {result.verdict}
                                </h1>
                                <p>
                                    CONFIDENCE INDEX:{" "}
                                    <b>{result.confidence}%</b>
                                </p>
                            </div>

                            <div style={styles.resultsGrid}>
                                <div style={styles.liveViewCard}>
                                    <div style={styles.scannerLine}></div>
                                    <video
                                        src={previewUrl}
                                        autoPlay
                                        loop
                                        muted
                                        style={styles.miniVideo}
                                    />
                                    <div style={styles.overlayText}>
                                        SCANNING_SIGNAL_HASH...
                                    </div>
                                </div>

                                <div style={styles.metricsContainer}>
                                    <h4 style={styles.metricsTitle}>
                                        SIGNAL DIAGNOSTICS
                                    </h4>
                                    <MetricBar
                                        label="Spatial Consistency"
                                        score={result.branch_scores?.spatial || 0}
                                    />
                                    <MetricBar
                                        label="Physical Integrity"
                                        score={result.branch_scores?.physics || 0}
                                    />
                                    <MetricBar
                                        label="Digital Forensics"
                                        score={result.branch_scores?.forensics || 0}
                                    />
                                    <MetricBar
                                        label="Audio Biometrics"
                                        score={result.branch_scores?.audio || 0}
                                    />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <style>{`
                @keyframes slideUp {
                    from { transform: translateY(40px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }

                @keyframes scan {
                    0% { top: 0; }
                    100% { top: 100%; }
                }

                .pulse-glow {
                    animation: glowAnim 1.5s infinite;
                }

                @keyframes glowAnim {
                    0% { box-shadow: 0 0 0 0 rgba(56,189,248,0.5); }
                    70% { box-shadow: 0 0 0 20px rgba(56,189,248,0); }
                    100% { box-shadow: 0 0 0 0 rgba(56,189,248,0); }
                }

                body::before {
                    content: "";
                    position: fixed;
                    width: 100%;
                    height: 100%;
                    background-image:
                        linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px);
                    background-size: 60px 60px;
                    pointer-events: none;
                    z-index: 0;
                }
            `}</style>
        </div>
    );
};

const MetricBar = ({ label, score }) => (
    <div style={{ marginBottom: "20px" }}>
        <div
            style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "6px",
            }}
        >
            <span style={{ fontSize: "12px", color: "#94a3b8" }}>
                {label}
            </span>
            <span
                style={{
                    fontSize: "12px",
                    color: "#f8fafc",
                    fontWeight: "bold",
                }}
            >
                {score}%
            </span>
        </div>

        <div
            style={{
                height: "6px",
                backgroundColor: "#0f172a",
                borderRadius: "3px",
                border: "1px solid #1e293b",
            }}
        >
            <div
                style={{
                    height: "100%",
                    width: `${score}%`,
                    background:
                        score > 75
                            ? "linear-gradient(90deg,#ef4444,#b91c1c)"
                            : "linear-gradient(90deg,#38bdf8,#0ea5e9)",
                    borderRadius: "3px",
                    transition:
                        "width 1.2s cubic-bezier(.17,.67,.83,.67)",
                }}
            ></div>
        </div>
    </div>
);

const styles = {
    pageWrapper: {
        display: "flex",
        height: "100vh",
        backgroundColor: "#000814",
        overflow: "hidden",
    },

    mainArea: {
        flex: 1,
        marginLeft: "260px",
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
        background: `
            radial-gradient(circle at 20% 30%, rgba(56,189,248,0.08), transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(16,185,129,0.06), transparent 40%),
            linear-gradient(180deg, #020617 0%, #000814 100%)
        `,
    },

    contentPadding: { padding: "50px" },

    welcomeHeader: {
        display: "flex",
        justifyContent: "space-between",
        marginBottom: "50px",
    },

    mainTitle: {
        margin: 0,
        color: "#f8fafc",
        fontSize: "2.3rem",
        fontWeight: "800",
    },

    subTitle: { color: "#64748b", fontSize: "13px" },

    liveBadge: {
        padding: "8px 18px",
        backgroundColor: "rgba(6, 78, 59, 0.3)",
        color: "#10b981",
        borderRadius: "30px",
        fontSize: "11px",
        fontWeight: "bold",
        border: "1px solid rgba(16, 185, 129, 0.2)",
    },

    uploadTerminal: {
        background: "rgba(15, 23, 42, 0.6)",
        padding: "35px",
        borderRadius: "28px",
        border: "1px solid rgba(56, 189, 248, 0.15)",
        backdropFilter: "blur(18px)",
        boxShadow: "0 0 40px rgba(56,189,248,0.08)",
    },

    dropZone: {
        border: "2px dashed #334155",
        padding: "50px",
        textAlign: "center",
        borderRadius: "20px",
        backgroundColor: "rgba(15, 23, 42, 0.4)",
    },

    fileButton: {
        backgroundColor: "#38bdf8",
        color: "#001018",
        padding: "12px 25px",
        borderRadius: "8px",
        fontWeight: "800",
        fontSize: "11px",
        cursor: "pointer",
        display: "inline-block",
        marginTop: "20px",
    },

    previewContainer: {
        textAlign: "center",
        backgroundColor: "#000",
        borderRadius: "20px",
        padding: "15px",
        border: "1px solid #334155",
    },

    videoHeader: {
        color: "#38bdf8",
        fontSize: "11px",
        letterSpacing: "2px",
        marginBottom: "12px",
        textAlign: "left",
    },

    videoElement: {
        width: "100%",
        maxHeight: "320px",
        borderRadius: "10px",
    },

    changeBtn: {
        marginTop: "10px",
        backgroundColor: "transparent",
        color: "#64748b",
        border: "none",
        cursor: "pointer",
        fontSize: "11px",
        textDecoration: "underline",
    },

    scanBtn: (disabled) => ({
        width: "100%",
        padding: "18px",
        borderRadius: "14px",
        border: disabled ? "1px solid #1e293b" : "1px solid #38bdf8",
        fontSize: "14px",
        fontWeight: "900",
        letterSpacing: "1px",
        background: disabled
            ? "#0f172a"
            : "linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%)",
        color: disabled ? "#334155" : "#001018",
        cursor: disabled ? "not-allowed" : "pointer",
        marginTop: "25px",
        transition: "all 0.3s ease",
        boxShadow: disabled
            ? "none"
            : "0 0 25px rgba(56,189,248,0.3)",
    }),

    verdictPanel: (verdict) => ({
        padding: "40px",
        borderRadius: "28px",
        textAlign: "center",
        color: "white",
        background:
            verdict === "FAKE"
                ? "linear-gradient(135deg, rgba(127,29,29,0.9), rgba(69,10,10,0.95))"
                : "linear-gradient(135deg, rgba(6,78,59,0.9), rgba(2,44,34,0.95))",
        border: `1px solid ${
            verdict === "FAKE"
                ? "rgba(239,68,68,0.6)"
                : "rgba(16,185,129,0.6)"
        }`,
        boxShadow:
            verdict === "FAKE"
                ? "0 0 40px rgba(239,68,68,0.25)"
                : "0 0 40px rgba(16,185,129,0.25)",
    }),

    verdictText: {
        margin: 0,
        fontSize: "3.8rem",
        fontWeight: "900",
    },

    resultsGrid: {
        display: "grid",
        gridTemplateColumns: "1.2fr 1fr",
        gap: "30px",
        marginTop: "30px",
    },

    liveViewCard: {
        position: "relative",
        backgroundColor: "#000",
        borderRadius: "28px",
        overflow: "hidden",
        border: "1px solid #38bdf844",
    },

    miniVideo: {
        width: "100%",
        height: "100%",
        objectFit: "cover",
        opacity: 0.6,
    },

    scannerLine: {
        position: "absolute",
        width: "100%",
        height: "2px",
        background: "#38bdf8",
        boxShadow: "0 0 15px #38bdf8",
        zIndex: 10,
        animation: "scan 2s linear infinite",
    },

    overlayText: {
        position: "absolute",
        bottom: "15px",
        left: "15px",
        color: "#38bdf8",
        fontSize: "10px",
        fontFamily: "monospace",
    },

    metricsContainer: {
        padding: "35px",
        background: "rgba(2, 6, 23, 0.8)",
        borderRadius: "28px",
        border: "1px solid rgba(56,189,248,0.08)",
        boxShadow: "inset 0 0 20px rgba(56,189,248,0.05)",
    },

    metricsTitle: {
        color: "#38bdf8",
        fontSize: "12px",
        marginBottom: "25px",
        letterSpacing: "1px",
    },
};

export default Dashboard;