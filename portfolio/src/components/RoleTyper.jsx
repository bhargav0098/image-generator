import { useState, useEffect, useCallback } from "react";

/**
 * Typewriter effect for cycling through role titles
 */
export default function RoleTyper({ roles }) {
  const [text, setText] = useState("");
  const [roleIdx, setRoleIdx] = useState(0);
  const [charIdx, setCharIdx] = useState(0);
  const [deleting, setDeleting] = useState(false);

  const tick = useCallback(() => {
    const word = roles[roleIdx];
    if (!deleting) {
      if (charIdx <= word.length) {
        setText(word.slice(0, charIdx));
        setCharIdx((c) => c + 1);
        if (charIdx === word.length) {
          setTimeout(() => setDeleting(true), 1600);
          return;
        }
      }
    } else {
      if (charIdx >= 0) {
        setText(word.slice(0, charIdx));
        setCharIdx((c) => c - 1);
        if (charIdx === 0) {
          setDeleting(false);
          setRoleIdx((r) => (r + 1) % roles.length);
          return;
        }
      }
    }
  }, [charIdx, deleting, roleIdx, roles]);

  useEffect(() => {
    const speed = deleting ? 35 : 65;
    const timer = setTimeout(tick, speed);
    return () => clearTimeout(timer);
  }, [tick, deleting]);

  return (
    <div className="hero-role" aria-label={`Role: ${roles[roleIdx]}`}>
      <span className="role-prefix">// </span>
      <span className="role-text">{text}</span>
      <span className="role-caret" aria-hidden="true">|</span>
    </div>
  );
}
