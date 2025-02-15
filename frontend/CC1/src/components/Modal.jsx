import { useState } from 'react';

const RoadmapModal = ({ isOpen, onClose, onSubmit }) => {
  const [startupName, setStartupName] = useState('');
  const [timeline, setTimeline] = useState(12);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({ startupName, timeline });
    setStartupName('');
    setTimeline(12);
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>Generate Roadmap</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Startup Name"
            value={startupName}
            onChange={(e) => setStartupName(e.target.value)}
            required
          />
          <input
            type="number"
            min="1"
            max="60"
            value={timeline}
            onChange={(e) => setTimeline(Number(e.target.value))}
          />
          <div className="modal-buttons">
            <button type="submit">Generate</button>
            <button type="button" onClick={onClose}>Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default RoadmapModal;