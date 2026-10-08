"use strict";

const labels = {siren: "Siren", glass_break: "Glass break", gun_shot: "Gun shot", baby_cry: "Baby cry", screaming: "Screaming", airport: "Airport", metro_station: "Metro station", public_square: "Public square"};
const state = {data: null, kind: "novel", selected: {novel: 0, normal: 0}};
const formatScore = value => value === null ? "N/A" : Number(value).toPrecision(2);
const titleCase = text => text.charAt(0).toUpperCase() + text.slice(1).replaceAll("_", " ");

function element(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}

function link(text, url) {
  const node = element("a", "", text);
  node.href = url;
  return node;
}

function trackCard(track, sample, reference) {
  const card = element("article", `track${track.key === "anomsep" ? " ours" : ""}`);
  const header = element("div", "track-header");
  const heading = element("h4", "", track.name);
  if (!reference) heading.append(element("span", "component", "Estimated anomalous component"));
  header.append(heading);
  const spec = element("div", "spec");
  const img = element("img");
  img.src = track.spectrogram;
  img.alt = `${track.name} spectrogram, shared intensity scale for this example`;
  img.width = 574;
  img.height = 231;
  img.loading = "lazy";
  spec.append(img);
  const audio = element("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.src = track.audio;
  audio.setAttribute("aria-label", `Play ${track.name}: ${sample.kind === "novel" ? labels[sample.novel_class] : "normal sounds"} in ${labels[sample.scene]}`);
  audio.append(link("Download audio", track.audio));
  audio.addEventListener("play", () => {
    document.querySelectorAll("audio").forEach(other => { if (other !== audio) other.pause(); });
  });
  card.append(header, spec, audio);
  if (reference) {
    const notes = {mixture: "Input to every method", normal: "Ground-truth normal component", novel: sample.kind === "normal" ? "Ground truth: silence" : "Ground-truth anomalous component"};
    card.append(element("div", "reference-note", notes[track.key]));
  } else {
    const metrics = element("div", "metrics");
    for (const [name, value] of [["CLAP ↑", track.clap], ["SAJ ↑", track.saj]]) {
      const metric = element("div", "", name);
      metric.append(element("strong", "", formatScore(value)));
      metrics.append(metric);
    }
    const correct = track.predicted_label === (sample.kind === "novel" ? 1 : 0);
    const decision = element("div", `decision${correct ? "" : " incorrect"}`, "Detection: ");
    decision.append(element("span", "", track.predicted_label === 1 ? "Anomalous" : "Normal"), document.createTextNode(correct ? " · correct" : " · incorrect"));
    card.append(metrics, decision);
  }
  return card;
}

function renderCredits(sample) {
  const details = element("details", "credits");
  details.append(element("summary", "", "Source recordings and credits"));
  const list = element("ul");
  for (const credit of sample.attributions) {
    const item = element("li");
    item.append(element("code", "", credit.file), document.createTextNode(` — ${credit.dataset}; ${credit.creator}. `), link("Source", credit.url), document.createTextNode(" · "), link(credit.license, credit.license_url));
    if (credit.title) item.append(document.createTextNode(` Original title: ${credit.title}.`));
    if (credit.provenance) item.append(document.createTextNode(` ${credit.provenance}. The dataset license is the uploader’s declaration.`));
    list.append(item);
  }
  details.append(list, element("p", "", `Evaluation sample: ${sample.id}. Sources were cropped, resampled, normalized, mixed, and processed by the separation methods. All concatenated source recordings are credited above. Shared playback gain: ${formatScore(sample.playback_gain)}.`));
  return details;
}

function renderSample(sample) {
  document.querySelectorAll("audio").forEach(audio => audio.pause());
  const container = document.getElementById("sample");
  container.replaceChildren();
  const header = element("div", "sample-header");
  const text = element("div");
  text.append(element("h3", "", sample.kind === "novel" ? `${labels[sample.novel_class]} in ${labels[sample.scene].toLowerCase()}` : `${labels[sample.scene]} · normal only`));
  const location = sample.kind === "novel" ? `${titleCase(sample.city)} · ` : "";
  text.append(element("p", "", `${location}${formatScore(sample.duration)} s · ${sample.kind === "novel" ? "Anomalous sound present" : "No anomalous sound present"}`));
  header.append(text);
  container.append(header, element("p", "track-label", "Reference audio"));
  const references = element("div", "tracks reference-tracks");
  sample.tracks.slice(0, 3).forEach(track => references.append(trackCard(track, sample, true)));
  container.append(references, element("p", "track-label", "Estimated anomalous sounds"));
  const methods = element("div", "tracks method-tracks");
  sample.tracks.slice(3).forEach(track => methods.append(trackCard(track, sample, false)));
  container.append(methods, renderCredits(sample));
}

function render() {
  const examples = state.data.samples.filter(sample => sample.kind === state.kind);
  const picker = document.getElementById("sample-picker");
  picker.replaceChildren();
  examples.forEach((sample, index) => {
    const button = element("button");
    button.type = "button";
    button.setAttribute("aria-pressed", String(index === state.selected[state.kind]));
    button.append(element("strong", "", state.kind === "novel" ? labels[sample.novel_class] : labels[sample.scene]));
    button.append(element("span", "", state.kind === "novel" ? labels[sample.scene] : `Example ${index + 1}`));
    button.addEventListener("click", () => {
      state.selected[state.kind] = index;
      render();
      document.querySelectorAll("#sample-picker button")[index].focus();
    });
    picker.append(button);
  });
  renderSample(examples[state.selected[state.kind]]);
  document.getElementById("example-panel").setAttribute("aria-labelledby", `tab-${state.kind}`);
  document.querySelectorAll("[data-kind]").forEach(button => {
    const active = button.dataset.kind === state.kind;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
}

document.querySelectorAll("[data-kind]").forEach(button => {
  button.addEventListener("click", () => { if (state.data) { state.kind = button.dataset.kind; render(); } });
  button.addEventListener("keydown", event => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key) && state.data) {
      event.preventDefault();
      state.kind = event.key === "Home" ? "novel" : event.key === "End" ? "normal" : state.kind === "novel" ? "normal" : "novel";
      render();
      document.getElementById(`tab-${state.kind}`).focus();
    }
  });
});

fetch("assets/project-data.json?v=9eff4d5d").then(response => {
  if (!response.ok) throw new Error("Unable to load the example manifest");
  return response.json();
}).then(data => { state.data = data; render(); }).catch(() => {
  document.getElementById("sample").replaceChildren(element("p", "loading", "The listening examples could not be loaded. Please reload this page, or browse the audio files in the GitHub repository."));
});
