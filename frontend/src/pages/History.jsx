import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Agar axios install nahi hai toh 'npm install axios' karlo

const History = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    // Database se data lene ka function
    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:8000/api/history');
                setHistory(response.data);
                setLoading(false);
            } catch (error) {
                console.error("Database fetch error:", error);
                setLoading(false);
            }
        };
        fetchHistory();
    }, []);

    return (
        <div style={{ padding: '40px', animation: "fadeIn 1s ease" }}>
            <h2 style={{ color: '#38bdf8', marginBottom: '30px', textTransform: 'uppercase', letterSpacing: '2px' }}>
                📂 Global Governance Logs
            </h2>
            
            {loading ? (
                <div style={{ textAlign: 'center', color: '#38bdf8', marginTop: '50px' }}>
                    <div className="spinner"></div>
                    <p>Accessing SQL Server Records...</p>
                </div>
            ) : (
                <div style={styles.tableContainer}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', color: '#cbd5e1' }}>
                        <thead>
                            <tr style={{ backgroundColor: '#0f172a', textAlign: 'left' }}>
                                <th style={styles.th}>Source File</th>
                                <th style={styles.th}>AI Verdict</th>
                                <th style={styles.th}>Confidence</th>
                                <th style={styles.th}>Processing Time</th>
                                <th style={styles.th}>Timestamp</th>
                            </tr>
                        </thead>
                        <tbody>
                            {history.length > 0 ? history.map((item, index) => (
                                <tr key={item.analysis_id} style={{ 
                                    borderBottom: '1px solid #334155',
                                    backgroundColor: index % 2 === 0 ? 'transparent' : 'rgba(15, 23, 42, 0.3)'
                                }} className="table-row">
                                    <td style={styles.td}>💾 {item.filename}</td>
                                    <td style={{ 
                                        ...styles.td, 
                                        color: item.verdict === "FAKE" ? "#f87171" : "#4ade80",
                                        fontWeight: 'bold'
                                    }}>
                                        {item.verdict}
                                    </td>
                                    <td style={styles.td}>{item.confidence_score}%</td>
                                    <td style={styles.td}>{item.processing_time_sec}s</td>
                                    <td style={styles.td}>
                                        {new Date(item.timestamp).toLocaleString()}
                                    </td>
                                </tr>
                            )) : (
                                <tr>
                                    <td colSpan="5" style={{textAlign: 'center', padding: '20px'}}>No records found in database.</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            )}

            <style>{`
                .spinner {
                    width: 40px; height: 40px; border: 4px solid #1e293b;
                    border-top: 4px solid #38bdf8; border-radius: 50%;
                    animation: spin 1s linear infinite; margin: 0 auto 20px;
                }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                .table-row:hover { background-color: rgba(56, 189, 248, 0.1) !important; }
            `}</style>
        </div>
    );
};

const styles = {
    tableContainer: { 
        overflow: 'hidden', borderRadius: '15px', border: '1px solid #334155',
        background: 'rgba(30, 41, 59, 0.7)', backdropFilter: 'blur(10px)'
    },
    th: { padding: '15px 20px', color: '#38bdf8', fontSize: '14px', textTransform: 'uppercase' },
    td: { padding: '15px 20px', fontSize: '15px' }
};

export default History;