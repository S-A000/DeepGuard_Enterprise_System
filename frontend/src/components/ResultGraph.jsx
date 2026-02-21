import React from 'react';

const ResultGraph = ({ scores }) => {
    return (
        <div style={{ marginTop: '20px', backgroundColor: '#0f172a', padding: '20px', borderRadius: '10px' }}>
            <h4 style={{ margin: '0 0 15px 0', color: '#38bdf8' }}>📊 Modality Detection Confidence:</h4>
            {Object.entries(scores).map(([key, value]) => (
                <div key={key} style={{ marginBottom: '15px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px', color: '#94a3b8' }}>
                        <span style={{ textTransform: 'capitalize' }}>{key}</span>
                        <span>{value}%</span>
                    </div>
                    <div style={{ height: '8px', backgroundColor: '#334155', borderRadius: '4px' }}>
                        <div style={{ 
                            width: `${value}%`, height: '100%', borderRadius: '4px',
                            backgroundColor: value > 60 ? '#ef4444' : '#22c55e',
                            transition: 'width 1s ease-in-out'
                        }}></div>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default ResultGraph;