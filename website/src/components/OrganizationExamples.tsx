import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { CollectionEntry } from "astro:content";
import "../styles/organization-examples.css";

type Example = CollectionEntry<"examples">["data"];
type Role = Example["roles"][number];
type Scene = Example["scenes"][number];
type AgentCopy = Scene["copies"][number];

function CopyRows({ copies }: { copies: AgentCopy[] }) {
  return copies.map((copy) => (
    <div className="example-copy" key={copy.id} data-active={copy.active}>
      <span>{copy.id}</span>
      <span>{copy.status}</span>
    </div>
  ));
}

function OrganizationRole({ role, scene }: { role: Role; scene: Scene }) {
  return (
    <div className={role.independent ? "example-independent" : undefined}>
      <div
        className="example-role"
        data-active={scene.active.includes(role.id)}
      >
        <div className="example-role-heading">
          <span className="example-role-name">
            {role.name}
            {role.annotation && (
              <span className="example-annotation"> / {role.annotation}</span>
            )}
          </span>
          <span className="example-role-state">
            {scene.states[role.id] ?? role.status}
          </span>
        </div>
        <div className="example-duty">{role.duty}</div>
      </div>
      {role.coverage && (
        <div
          className="example-coverage"
          role="group"
          aria-label="Ongoing sector research"
        >
          <CopyRows copies={scene.coverage ?? []} />
        </div>
      )}
      {role.copies && (
        <div className="example-copies">
          <div className="example-copy-label">{scene.copyLabel}</div>
          <CopyRows copies={scene.copies} />
        </div>
      )}
      {role.children && (
        <ul className="example-branches">
          {role.children.map((child) => (
            <li key={child.id}>
              <OrganizationRole role={child} scene={scene} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ClinicalDraft({
  pause,
  draft,
}: {
  pause: () => void;
  draft: NonNullable<Example["draft"]>;
}) {
  return (
    <details
      className="example-draft"
      onToggle={(event) => {
        if (event.currentTarget.open) pause();
      }}
    >
      <summary>[review draft]</summary>
      <div>
        <p>
          <strong>{draft.title}</strong>
          <br />
          <span className="muted">{draft.note}</span>
        </p>
        {draft.sections.map((section) => (
          <p key={section.title}>
            <strong>{section.title}</strong>
            <br />
            {section.text}
          </p>
        ))}
      </div>
    </details>
  );
}

function ExamplePlayer({
  example,
  active,
}: {
  example: Example;
  active: boolean;
}) {
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const [inView, setInView] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);
  const panel = useRef<HTMLDivElement>(null);
  const sceneDelay = (step: number) =>
    (example.scenes[step + 1]?.at ?? example.duration) -
    example.scenes[step].at;
  const remaining = useRef(sceneDelay(0));
  const scene = example.scenes[index];

  useEffect(() => {
    const preference = matchMedia("(prefers-reduced-motion: reduce)");
    const syncMotion = () => setReducedMotion(preference.matches);
    const syncVisibility = () => setPageVisible(!document.hidden);
    syncMotion();
    syncVisibility();
    preference.addEventListener("change", syncMotion);
    document.addEventListener("visibilitychange", syncVisibility);
    const observer = new IntersectionObserver(
      (entries) => setInView(entries.some((entry) => entry.isIntersecting)),
      { threshold: 0.1 },
    );
    if (panel.current) observer.observe(panel.current);
    return () => {
      preference.removeEventListener("change", syncMotion);
      document.removeEventListener("visibilitychange", syncVisibility);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!active || paused || !inView || !pageVisible || reducedMotion) return;
    const started = performance.now();
    let completed = false;
    const timer = window.setTimeout(() => {
      completed = true;
      const next = (index + 1) % example.scenes.length;
      remaining.current =
        (example.scenes[next + 1]?.at ?? example.duration) -
        example.scenes[next].at;
      setIndex(next);
    }, remaining.current);
    return () => {
      window.clearTimeout(timer);
      if (!completed)
        remaining.current = Math.max(
          0,
          remaining.current - (performance.now() - started),
        );
    };
  }, [active, paused, inView, pageVisible, reducedMotion, index, example]);

  function advance() {
    const next = (index + 1) % example.scenes.length;
    remaining.current = sceneDelay(next);
    setIndex(next);
  }

  const control = reducedMotion
    ? index === example.scenes.length - 1
      ? "replay"
      : "next"
    : paused
      ? "resume"
      : "pause";
  return (
    <div
      ref={panel}
      role="tabpanel"
      id={`example-panel-${example.id}`}
      aria-labelledby={`example-tab-${example.id}`}
      hidden={!active}
      className="example-panel"
      tabIndex={0}
    >
      <figure>
        <figcaption>
          <span className="example-caption">
            <span>{example.filename}</span>
            <span>{example.note}</span>
          </span>
          <button
            type="button"
            className="example-playback"
            onClick={
              reducedMotion ? advance : () => setPaused((value) => !value)
            }
            aria-label={`${control} ${example.label} example`}
          >
            [{control}]
          </button>
        </figcaption>
        <div className="example-content">
          <div
            className="example-channels"
            role="group"
            aria-label="Communication channels"
          >
            {example.channels.map((channel) => (
              <span key={channel.id} data-active={scene.channel === channel.id}>
                [{channel.label}]
              </span>
            ))}
          </div>
          <div
            className="example-tree"
            role="group"
            aria-label="Reporting structure and standing responsibilities"
          >
            {example.roles.map((role) => (
              <OrganizationRole key={role.id} role={role} scene={scene} />
            ))}
          </div>
          <div
            className="example-event"
            aria-live={active && (paused || reducedMotion) ? "polite" : "off"}
            aria-atomic="true"
          >
            <div className="example-route">{scene.route}</div>
            <p>{scene.text}</p>
          </div>
          <div className="example-footer">
            <span>{scene.context}</span>
            {scene.gate && <span className="example-gate">{scene.gate}</span>}
          </div>
          {example.id === "medical" && scene.ready && example.draft && (
            <ClinicalDraft
              pause={() => setPaused(true)}
              draft={example.draft}
            />
          )}
        </div>
      </figure>
    </div>
  );
}

export function OrganizationExamples({
  examples,
}: {
  examples: Example[];
}): React.JSX.Element {
  const [selected, setSelected] = useState(0);
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);

  function navigate(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number;
    switch (event.key) {
      case "ArrowRight":
        next = (index + 1) % examples.length;
        break;
      case "ArrowLeft":
        next = (index + examples.length - 1) % examples.length;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = examples.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    setSelected(next);
    tabs.current[next]?.focus();
  }

  return (
    <div className="organization-examples">
      <div
        role="tablist"
        aria-label="Autonomous organization examples"
        className="example-tabs"
      >
        {examples.map((example, index) => (
          <button
            key={example.id}
            ref={(element) => {
              tabs.current[index] = element;
            }}
            type="button"
            role="tab"
            id={`example-tab-${example.id}`}
            aria-controls={`example-panel-${example.id}`}
            aria-selected={selected === index}
            tabIndex={selected === index ? 0 : -1}
            onClick={() => setSelected(index)}
            onKeyDown={(event) => navigate(event, index)}
          >
            {example.label}
          </button>
        ))}
      </div>
      {examples.map((example, index) => (
        <ExamplePlayer
          key={example.id}
          example={example}
          active={selected === index}
        />
      ))}
    </div>
  );
}
