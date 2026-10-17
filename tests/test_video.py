import numpy as np

from src.config import VIDEO_DIR
from src.datasets.generate_video import FPS, frontal_pose, sagittal_pose
from src.video.movement_analysis import analyze_shoulder_abduction, analyze_squat, detect_fall, shoulder_abduction
from src.video.objects import SidecarDetector, inside
from src.video.pose import PoseSequence, load_sidecar, to_openpose18
from src.video.procedure_analysis import analyze_surgery


def _seq(poses):
    kp = np.array([np.hstack([p, np.ones((17, 1))]) for p in poses])
    return PoseSequence(kp, FPS, "teste")


def test_shoulder_angle_geometry():
    seq = _seq([frontal_pose(90, 30)])
    assert abs(shoulder_abduction(seq, "esq")[0] - 90) < 1
    assert abs(shoulder_abduction(seq, "dir")[0] - 30) < 1


def test_openpose18_mapping_adds_neck():
    kp = np.hstack([frontal_pose(0, 0), np.ones((17, 1))])
    op = to_openpose18(kp)
    assert op.shape == (18, 3)
    assert np.allclose(op[1, :2], (kp[5, :2] + kp[6, :2]) / 2)  # pescoço
    assert np.allclose(op[2], kp[6])                            # ombro direito


def test_normal_session_has_no_findings():
    seq, _ = load_sidecar(VIDEO_DIR / "PAC-0001_fisio_ombro.mp4")
    r = analyze_shoulder_abduction(seq)
    assert r.n_reps == 5 and r.findings == []


def test_hemiparesis_detected():
    seq, _ = load_sidecar(VIDEO_DIR / "PAC-0003_fisio_ombro.mp4")
    kinds = {f.kind for f in analyze_shoulder_abduction(seq).findings}
    assert {"assimetria_membros_superiores", "adm_reduzida_ombro_esquerdo", "compensacao_tronco"} <= kinds


def test_fall_detection():
    standing = [sagittal_pose(0)] * 60
    falling = [sagittal_pose(0, fall=min(1, i / 24) ** 2) for i in range(60)]
    assert detect_fall(_seq(standing)) is None
    assert detect_fall(_seq(standing + falling)) is not None
    seq, _ = load_sidecar(VIDEO_DIR / "PAC-0005_fisio_agachamento.mp4")
    assert any(f.kind == "queda_detectada" and f.severity == "critico" for f in analyze_squat(seq).findings)


def test_surgery_count_and_zones():
    det = SidecarDetector(VIDEO_DIR / "PAC-0006_cirurgia.mp4")
    r = analyze_surgery(det.frames, det.fps, det.meta["zonas"])
    kinds = {f.kind for f in r.findings}
    assert {"falha_contagem_instrumental", "instrumento_fora_campo_esteril", "quebra_area_esteril"} <= kinds
    count = next(f for f in r.findings if f.kind == "falha_contagem_instrumental")
    assert count.evidence["faltantes"].get("tesoura") == 1
    assert "tesoura na área da incisão" in count.evidence["localizados"]


def test_inside_zone():
    assert inside([10, 10, 20, 20], (0, 0, 30, 30))
    assert not inside([40, 40, 50, 50], (0, 0, 30, 30))
