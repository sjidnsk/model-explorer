from __future__ import annotations
from collections import Counter
from typing import Any
from .path_feedback_diagnostic_aggregate import GCS_CONTROL_POINT_BACKEND, PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES, _aggregate_metric_stats, _counter_dict, _first_float, _first_present, _float_value, _int_value, _numeric_metric_stats

def _iris_diagnostics(evaluations) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    report_count = 0
    fallback_count = 0
    failure_count = 0
    region_count_total = 0
    for item in evaluations:
        iris = item.to_dict().get('iris_region')
        if not isinstance(iris, dict):
            continue
        report_count += 1
        status = str(iris.get('status') or 'unknown')
        status_counts[status] += 1
        region_count_total += _int_value(iris.get('region_count'))
        if bool(iris.get('fallback_used')):
            fallback_count += 1
        if status == 'failed':
            failure_count += 1
        reason = iris.get('failure_reason')
        if reason:
            fallback_reasons[str(reason)] += 1
    return {'report_count': report_count, 'status_counts': dict(sorted(status_counts.items())), 'fallback_count': fallback_count, 'failure_count': failure_count, 'region_count_total': region_count_total, 'fallback_reasons': dict(sorted(fallback_reasons.items()))}

def _region_graph_diagnostics(evaluations) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    fallback_count = 0
    start_goal_disconnected_count = 0
    for item in evaluations:
        graph = item.to_dict().get('region_graph')
        if not isinstance(graph, dict):
            continue
        source_counts[str(graph.get('graph_source') or graph.get('region_source') or 'unknown')] += 1
        if bool(graph.get('fallback_used')):
            fallback_count += 1
        if graph.get('start_goal_connected') is False:
            start_goal_disconnected_count += 1
        reason = graph.get('fallback_reason')
        if reason:
            fallback_reasons[str(reason)] += 1
    return {'source_counts': dict(sorted(source_counts.items())), 'fallback_count': fallback_count, 'fallback_reasons': dict(sorted(fallback_reasons.items())), 'start_goal_disconnected_count': start_goal_disconnected_count}

def _convex_region_diagnostics(evaluations) -> dict[str, Any]:
    backend_counts: Counter[str] = Counter()
    coverage_status_counts: Counter[str] = Counter()
    gcs_ready_reason_counts: Counter[str] = Counter()
    report_count = 0
    region_count_total = 0
    fallback_used_count = 0
    gcs_ready_count = 0
    blocked_cell_violation_count = 0
    start_contained_count = 0
    goal_contained_count = 0
    adjacent_overlap_count = 0
    portal_count = 0
    for item in evaluations:
        convex = item.to_dict().get('convex_region')
        if not isinstance(convex, dict):
            continue
        report_count += 1
        backend = convex.get('backend')
        if backend:
            backend_counts[str(backend)] += 1
        coverage_status = convex.get('coverage_status')
        if coverage_status:
            coverage_status_counts[str(coverage_status)] += 1
        reason = convex.get('gcs_ready_reason')
        if reason:
            gcs_ready_reason_counts[str(reason)] += 1
        region_count_total += _int_value(convex.get('region_count'))
        if convex.get('fallback_used') is True:
            fallback_used_count += 1
        if convex.get('gcs_ready') is True:
            gcs_ready_count += 1
        if convex.get('start_contained') is True:
            start_contained_count += 1
        if convex.get('goal_contained') is True:
            goal_contained_count += 1
        blocked_cell_violation_count += _int_value(convex.get('blocked_cell_violation_count'))
        adjacent_overlap_count += _int_value(convex.get('adjacent_overlap_count'))
        portal_count += _int_value(convex.get('portal_count'))
    return {'report_count': report_count, 'region_count_total': region_count_total, 'backend_counts': dict(sorted(backend_counts.items())), 'fallback_used_count': fallback_used_count, 'gcs_ready_count': gcs_ready_count, 'blocked_cell_violation_count': blocked_cell_violation_count, 'coverage_status_counts': dict(sorted(coverage_status_counts.items())), 'gcs_ready_reason_counts': dict(sorted(gcs_ready_reason_counts.items())), 'start_contained_count': start_contained_count, 'goal_contained_count': goal_contained_count, 'adjacent_overlap_count': adjacent_overlap_count, 'portal_count': portal_count}

def _gcs_trajectory_diagnostics(evaluations) -> dict[str, Any]:
    backend_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    result_status_counts: Counter[str] = Counter()
    report_count = 0
    attempted_count = 0
    success_count = 0
    collision_count = 0
    region_count_total = 0
    sample_count_total = 0
    for item in evaluations:
        gcs = item.to_dict().get('gcs_trajectory')
        if not isinstance(gcs, dict):
            continue
        report_count += 1
        backend = gcs.get('backend')
        if backend:
            backend_counts[str(backend)] += 1
        reason = gcs.get('reason')
        if reason:
            reason_counts[str(reason)] += 1
        result_status = gcs.get('result_status')
        if result_status:
            result_status_counts[str(result_status)] += 1
        if gcs.get('attempted') is True:
            attempted_count += 1
        if gcs.get('success') is True:
            success_count += 1
        collision_count += _int_value(gcs.get('collision_count'))
        region_count_total += _int_value(gcs.get('region_count'))
        sample_count_total += _int_value(gcs.get('sample_count'))
    return {'report_count': report_count, 'attempted_count': attempted_count, 'success_count': success_count, 'collision_count': collision_count, 'region_count_total': region_count_total, 'sample_count_total': sample_count_total, 'backend_counts': dict(sorted(backend_counts.items())), 'reason_counts': dict(sorted(reason_counts.items())), 'result_status_counts': dict(sorted(result_status_counts.items()))}

def _gcs_candidate_diagnostics(evaluations) -> dict[str, Any]:
    fallback_reason_counts: Counter[str] = Counter()
    selection_reason_counts: Counter[str] = Counter()
    report_count = 0
    attempted_count = 0
    available_count = 0
    selected_count = 0
    collision_count = 0
    delta_negative_count = 0
    delta_positive_count = 0
    delta_zero_count = 0
    for item in evaluations:
        candidate = item.to_dict().get('gcs_candidate')
        if not isinstance(candidate, dict):
            continue
        report_count += 1
        if candidate.get('attempted') is True:
            attempted_count += 1
        if candidate.get('available') is True:
            available_count += 1
        if candidate.get('selected') is True:
            selected_count += 1
        collision_count += _int_value(candidate.get('collision_count'))
        fallback_reason = candidate.get('fallback_reason')
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1
        selection_reason = candidate.get('selection_reason')
        if selection_reason:
            selection_reason_counts[str(selection_reason)] += 1
        delta = _float_value(candidate.get('cost_delta_vs_baseline'))
        if delta is None:
            continue
        if delta < 0.0:
            delta_negative_count += 1
        elif delta > 0.0:
            delta_positive_count += 1
        else:
            delta_zero_count += 1
    return {'report_count': report_count, 'attempted_count': attempted_count, 'available_count': available_count, 'selected_count': selected_count, 'collision_count': collision_count, 'fallback_reason_counts': dict(sorted(fallback_reason_counts.items())), 'selection_reason_counts': dict(sorted(selection_reason_counts.items())), 'cost_delta_vs_baseline_negative_count': delta_negative_count, 'cost_delta_vs_baseline_positive_count': delta_positive_count, 'cost_delta_vs_baseline_zero_count': delta_zero_count}

def _gcs_control_point_diagnostics(evaluations) -> dict[str, Any]:
    backend_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    terrain_objective_source_counts: Counter[str] = Counter()
    report_count = 0
    attempted_count = 0
    success_count = 0
    candidate_selected_count = 0
    sampled_terrain_costs: list[float] = []
    high_cost_exposure_deltas: list[float] = []
    for item in evaluations:
        payload = item.to_dict()
        gcs = payload.get('gcs_trajectory')
        if not isinstance(gcs, dict) or gcs.get('backend') != GCS_CONTROL_POINT_BACKEND:
            continue
        report_count += 1
        backend_counts[str(gcs.get('backend'))] += 1
        if gcs.get('attempted') is True:
            attempted_count += 1
        if gcs.get('success') is True:
            success_count += 1
        candidate = payload.get('gcs_candidate')
        candidate = candidate if isinstance(candidate, dict) else {}
        if candidate.get('selected') is True:
            candidate_selected_count += 1
        fallback_reason = candidate.get('fallback_reason')
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1
        trajectory_cost = gcs.get('cost_summary')
        trajectory_cost = trajectory_cost if isinstance(trajectory_cost, dict) else {}
        candidate_cost = candidate.get('cost_summary')
        candidate_cost = candidate_cost if isinstance(candidate_cost, dict) else {}
        terrain_source = _first_present(trajectory_cost.get('terrain_objective_source'), candidate_cost.get('terrain_objective_source'))
        if terrain_source:
            terrain_objective_source_counts[str(terrain_source)] += 1
        sampled_terrain_cost = _first_float(trajectory_cost.get('sampled_terrain_cost'), candidate_cost.get('sampled_terrain_cost'))
        if sampled_terrain_cost is not None:
            sampled_terrain_costs.append(sampled_terrain_cost)
        exposure_delta = _first_float(candidate_cost.get('high_cost_exposure_delta_vs_baseline'), trajectory_cost.get('high_cost_exposure_delta_vs_baseline'))
        if exposure_delta is not None:
            high_cost_exposure_deltas.append(exposure_delta)
    sampled_stats = _numeric_metric_stats(sampled_terrain_costs)
    exposure_stats = _numeric_metric_stats(high_cost_exposure_deltas)
    return {'report_count': report_count, 'attempted_count': attempted_count, 'success_count': success_count, 'backend_counts': dict(sorted(backend_counts.items())), 'candidate_selected_count': candidate_selected_count, 'candidate_fallback_reason_counts': dict(sorted(fallback_reason_counts.items())), 'terrain_objective_source_counts': dict(sorted(terrain_objective_source_counts.items())), 'sampled_terrain_cost_count': sampled_stats['count'], 'sampled_terrain_cost_min': sampled_stats['min'], 'sampled_terrain_cost_max': sampled_stats['max'], 'sampled_terrain_cost_mean': sampled_stats['mean'], 'high_cost_exposure_delta_count': exposure_stats['count'], 'high_cost_exposure_delta_min': exposure_stats['min'], 'high_cost_exposure_delta_max': exposure_stats['max'], 'high_cost_exposure_delta_mean': exposure_stats['mean']}

def _gcs_motion_feasibility_diagnostics(evaluations) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    motion_model_counts: Counter[str] = Counter()
    report_count = 0
    evaluated_count = 0
    feasible_count = 0
    infeasible_count = 0
    diagnostic_only_count = 0
    curvature_violation_count = 0
    heading_violation_count = 0
    for item in evaluations:
        motion = item.to_dict().get('gcs_motion_feasibility')
        if not isinstance(motion, dict):
            continue
        report_count += 1
        if motion.get('evaluated') is True:
            evaluated_count += 1
        status = motion.get('feasibility_status')
        if status:
            status_counts[str(status)] += 1
        if status == 'feasible':
            feasible_count += 1
        elif status == 'infeasible':
            infeasible_count += 1
        elif status == 'diagnostic_only':
            diagnostic_only_count += 1
        fallback_reason = motion.get('fallback_reason')
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1
        motion_model = motion.get('motion_model')
        if motion_model:
            motion_model_counts[str(motion_model)] += 1
        curvature_violation_count += _int_value(motion.get('curvature_violation_count'))
        heading_violation_count += _int_value(motion.get('heading_violation_count'))
    return {'report_count': report_count, 'evaluated_count': evaluated_count, 'feasible_count': feasible_count, 'infeasible_count': infeasible_count, 'diagnostic_only_count': diagnostic_only_count, 'curvature_violation_count': curvature_violation_count, 'heading_violation_count': heading_violation_count, 'status_counts': dict(sorted(status_counts.items())), 'fallback_reason_counts': dict(sorted(fallback_reason_counts.items())), 'motion_model_counts': dict(sorted(motion_model_counts.items()))}

def _gcs_curvature_constrained_diagnostics(evaluations) -> dict[str, Any]:
    status_before_counts: Counter[str] = Counter()
    status_after_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    repair_strategy_counts: Counter[str] = Counter()
    report_count = 0
    attempted_count = 0
    available_count = 0
    selected_count = 0
    repair_success_count = 0
    infeasible_count = 0
    diagnostic_only_count = 0
    curvature_violation_count_before = 0
    curvature_violation_count_after = 0
    heading_violation_count_before = 0
    heading_violation_count_after = 0
    collision_count = 0
    region_containment_violation_count = 0
    for item in evaluations:
        candidate = item.to_dict().get('gcs_curvature_constrained_candidate')
        if not isinstance(candidate, dict):
            continue
        report_count += 1
        if candidate.get('attempted') is True:
            attempted_count += 1
        if candidate.get('available') is True:
            available_count += 1
        if candidate.get('selected') is True:
            selected_count += 1
        if candidate.get('repair_success') is True:
            repair_success_count += 1
        status_before = candidate.get('status_before')
        if status_before:
            status_before_counts[str(status_before)] += 1
        status_after = candidate.get('status_after')
        if status_after:
            status_after_counts[str(status_after)] += 1
        if status_after == 'infeasible':
            infeasible_count += 1
        elif status_after == 'diagnostic_only':
            diagnostic_only_count += 1
        repair_strategy = candidate.get('repair_strategy')
        if repair_strategy:
            repair_strategy_counts[str(repair_strategy)] += 1
        fallback_reason = candidate.get('fallback_reason')
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1
        curvature_violation_count_before += _int_value(candidate.get('curvature_violation_count_before'))
        curvature_violation_count_after += _int_value(candidate.get('curvature_violation_count_after'))
        heading_violation_count_before += _int_value(candidate.get('heading_violation_count_before'))
        heading_violation_count_after += _int_value(candidate.get('heading_violation_count_after'))
        collision_count += _int_value(candidate.get('collision_count'))
        region_containment_violation_count += _int_value(candidate.get('region_containment_violation_count'))
    return {'report_count': report_count, 'attempted_count': attempted_count, 'available_count': available_count, 'selected_count': selected_count, 'repair_success_count': repair_success_count, 'infeasible_count': infeasible_count, 'diagnostic_only_count': diagnostic_only_count, 'curvature_violation_count_before': curvature_violation_count_before, 'curvature_violation_count_after': curvature_violation_count_after, 'heading_violation_count_before': heading_violation_count_before, 'heading_violation_count_after': heading_violation_count_after, 'collision_count': collision_count, 'region_containment_violation_count': region_containment_violation_count, 'status_before_counts': dict(sorted(status_before_counts.items())), 'status_after_counts': dict(sorted(status_after_counts.items())), 'fallback_reason_counts': dict(sorted(fallback_reason_counts.items())), 'repair_strategy_counts': dict(sorted(repair_strategy_counts.items()))}

def _channel_aware_astar_diagnostics(evaluations, *, scenario_id: str) -> dict[str, Any]:
    requested_backend_counts: Counter[str] = Counter()
    selected_backend_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    blocker_class_counts: Counter[str] = Counter()
    platform_goal_class_counts: Counter[str] = Counter()
    path_cost_deltas: list[float] = []
    channel_cost_deltas: list[float] = []
    high_cost_exposure_deltas: list[float] = []
    candidate_audit: list[dict[str, Any]] = []
    report_count = 0
    selected_count = 0
    fallback_count = 0
    path_changed_count = 0
    platform_goal_contract_mismatch_count = 0
    platform_goal_anchor_available_count = 0
    platform_goal_unresolved_count = 0
    for item in evaluations:
        candidate = item.to_dict()
        planning_backend = candidate.get('planning_backend')
        if not isinstance(planning_backend, dict):
            continue
        requested_backend = planning_backend.get('requested_backend')
        selected_backend = planning_backend.get('selected_backend')
        if requested_backend != 'channel_aware_astar' and selected_backend != 'channel_aware_astar':
            continue
        report_count += 1
        requested_key = str(requested_backend or 'unknown')
        selected_key = str(selected_backend or 'unknown')
        status = str(planning_backend.get('status') or 'unknown')
        fallback_reason = planning_backend.get('fallback_reason')
        fallback_reason_text = str(fallback_reason) if fallback_reason else None
        requested_backend_counts[requested_key] += 1
        selected_backend_counts[selected_key] += 1
        status_counts[status] += 1
        selected = status == 'selected' or selected_backend == 'channel_aware_astar'
        fallback = status == 'fallback' or bool(fallback_reason_text) or (not selected)
        if selected:
            selected_count += 1
        if fallback:
            fallback_count += 1
        if fallback_reason_text:
            fallback_reason_counts[fallback_reason_text] += 1
        platform_goal_feasibility = candidate.get('platform_goal_feasibility')
        platform_goal_feasibility = platform_goal_feasibility if isinstance(platform_goal_feasibility, dict) else {}
        platform_goal_class = str(platform_goal_feasibility.get('classification') or 'unavailable')
        platform_goal_class_counts[platform_goal_class] += 1
        if _platform_goal_contract_mismatch(platform_goal_feasibility):
            platform_goal_contract_mismatch_count += 1
        if platform_goal_feasibility.get('nearest_inflated_passable_anchor') is not None:
            platform_goal_anchor_available_count += 1
        if platform_goal_class == 'unknown_contract_mismatch':
            platform_goal_unresolved_count += 1
        blocker_class = _channel_aware_astar_blocker_class(status=status, selected_backend=selected_key, fallback_reason=fallback_reason_text, platform_goal_feasibility=platform_goal_feasibility)
        failure_taxonomy = _channel_aware_astar_failure_taxonomy(blocker_class=blocker_class, fallback_reason=fallback_reason_text)
        blocker_class_counts[blocker_class] += 1
        comparison = planning_backend.get('comparison')
        comparison = comparison if isinstance(comparison, dict) else {}
        if comparison.get('path_changed') is True:
            path_changed_count += 1
        path_cost_delta = _float_value(comparison.get('path_cost_delta'))
        channel_cost_delta = _float_value(comparison.get('channel_cost_delta'))
        high_cost_delta = _float_value(comparison.get('high_cost_exposure_delta'))
        if path_cost_delta is not None:
            path_cost_deltas.append(path_cost_delta)
        if channel_cost_delta is not None:
            channel_cost_deltas.append(channel_cost_delta)
        if high_cost_delta is not None:
            high_cost_exposure_deltas.append(high_cost_delta)
        candidate_audit.append({'scenario_id': scenario_id, 'action_index': candidate.get('action_index'), 'cell': candidate.get('cell'), 'requested_backend': requested_key, 'selected_backend': selected_key, 'status': status, 'fallback_reason': fallback_reason_text, 'blocker_class': blocker_class, 'failure_taxonomy': failure_taxonomy, 'platform_goal_feasibility': platform_goal_feasibility, 'comparison': {'path_changed': comparison.get('path_changed'), 'path_cost_delta': path_cost_delta, 'channel_cost_delta': channel_cost_delta, 'high_cost_exposure_delta': high_cost_delta}})
    return {'report_count': report_count, 'selected_count': selected_count, 'fallback_count': fallback_count, 'requested_backend_counts': dict(sorted(requested_backend_counts.items())), 'selected_backend_counts': dict(sorted(selected_backend_counts.items())), 'status_counts': dict(sorted(status_counts.items())), 'fallback_reason_counts': dict(sorted(fallback_reason_counts.items())), 'blocker_class_counts': dict(sorted(blocker_class_counts.items())), 'platform_goal_feasibility_class_counts': dict(sorted(platform_goal_class_counts.items())), 'platform_goal_contract_mismatch_count': platform_goal_contract_mismatch_count, 'platform_goal_anchor_available_count': platform_goal_anchor_available_count, 'platform_goal_unresolved_count': platform_goal_unresolved_count, 'path_changed_count': path_changed_count, 'path_changed_rate': path_changed_count / report_count if report_count else 0.0, 'path_cost_delta': _numeric_metric_stats(path_cost_deltas), 'channel_cost_delta': _numeric_metric_stats(channel_cost_deltas), 'high_cost_exposure_delta': _numeric_metric_stats(high_cost_exposure_deltas), 'candidate_audit': candidate_audit}

def _channel_aware_astar_blocker_class(*, status: str, selected_backend: str, fallback_reason: str | None, platform_goal_feasibility: dict[str, Any] | None=None) -> str:
    if status == 'selected' or selected_backend == 'channel_aware_astar':
        return 'selected'
    reason = fallback_reason or ''
    if 'goal_blocked' in reason:
        platform_class = _platform_goal_failure_class(platform_goal_feasibility)
        if platform_class is not None:
            return platform_class
        return 'goal_blocked'
    if 'same_as_baseline' in reason:
        return 'same_as_baseline'
    if 'not_lower_risk' in reason:
        return 'not_lower_risk'
    if reason.startswith('channel_search_failed'):
        return 'search_failed'
    if reason:
        return reason
    return 'fallback_unspecified'

def _channel_aware_astar_failure_taxonomy(*, blocker_class: str, fallback_reason: str | None) -> str:
    if blocker_class in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
        return blocker_class
    if blocker_class in {'selected', 'same_as_baseline', 'not_lower_risk', 'goal_blocked'}:
        return blocker_class
    reason = fallback_reason or ''
    if 'goal_blocked' in reason:
        return 'goal_blocked'
    if reason.startswith('channel_search_failed'):
        return 'search_failed'
    return blocker_class

def _platform_goal_failure_class(feasibility: dict[str, Any] | None) -> str | None:
    if not isinstance(feasibility, dict):
        return None
    classification = str(feasibility.get('classification') or '')
    if classification in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES:
        return classification
    return None

def _platform_goal_contract_mismatch(feasibility: dict[str, Any]) -> bool:
    return bool(feasibility.get('contract_reachable')) and str(feasibility.get('classification') or '') in PLATFORM_GOAL_CONTRACT_MISMATCH_CLASSES

def _aggregate_channel_aware_astar_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    requested_backend_counts: Counter[str] = Counter()
    selected_backend_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    blocker_class_counts: Counter[str] = Counter()
    platform_goal_class_counts: Counter[str] = Counter()
    path_cost_stats: list[dict[str, Any]] = []
    channel_cost_stats: list[dict[str, Any]] = []
    high_cost_stats: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []
    report_count = 0
    selected_count = 0
    fallback_count = 0
    path_changed_count = 0
    platform_goal_contract_mismatch_count = 0
    platform_goal_anchor_available_count = 0
    platform_goal_unresolved_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        report_count += _int_value(item.get('report_count'))
        selected_count += _int_value(item.get('selected_count'))
        fallback_count += _int_value(item.get('fallback_count'))
        path_changed_count += _int_value(item.get('path_changed_count'))
        requested_backend_counts.update(_counter_dict(item.get('requested_backend_counts')))
        selected_backend_counts.update(_counter_dict(item.get('selected_backend_counts')))
        status_counts.update(_counter_dict(item.get('status_counts')))
        fallback_reason_counts.update(_counter_dict(item.get('fallback_reason_counts')))
        blocker_class_counts.update(_counter_dict(item.get('blocker_class_counts')))
        platform_goal_class_counts.update(_counter_dict(item.get('platform_goal_feasibility_class_counts')))
        platform_goal_contract_mismatch_count += _int_value(item.get('platform_goal_contract_mismatch_count'))
        platform_goal_anchor_available_count += _int_value(item.get('platform_goal_anchor_available_count'))
        platform_goal_unresolved_count += _int_value(item.get('platform_goal_unresolved_count'))
        if isinstance(item.get('path_cost_delta'), dict):
            path_cost_stats.append(item['path_cost_delta'])
        if isinstance(item.get('channel_cost_delta'), dict):
            channel_cost_stats.append(item['channel_cost_delta'])
        if isinstance(item.get('high_cost_exposure_delta'), dict):
            high_cost_stats.append(item['high_cost_exposure_delta'])
        audit = item.get('candidate_audit')
        if isinstance(audit, list):
            candidate_audit.extend((entry for entry in audit if isinstance(entry, dict)))
    return {'report_count': report_count, 'selected_count': selected_count, 'fallback_count': fallback_count, 'requested_backend_counts': dict(sorted(requested_backend_counts.items())), 'selected_backend_counts': dict(sorted(selected_backend_counts.items())), 'status_counts': dict(sorted(status_counts.items())), 'fallback_reason_counts': dict(sorted(fallback_reason_counts.items())), 'blocker_class_counts': dict(sorted(blocker_class_counts.items())), 'platform_goal_feasibility_class_counts': dict(sorted(platform_goal_class_counts.items())), 'platform_goal_contract_mismatch_count': platform_goal_contract_mismatch_count, 'platform_goal_anchor_available_count': platform_goal_anchor_available_count, 'platform_goal_unresolved_count': platform_goal_unresolved_count, 'path_changed_count': path_changed_count, 'path_changed_rate': path_changed_count / report_count if report_count else 0.0, 'path_cost_delta': _aggregate_metric_stats(path_cost_stats), 'channel_cost_delta': _aggregate_metric_stats(channel_cost_stats), 'high_cost_exposure_delta': _aggregate_metric_stats(high_cost_stats), 'candidate_audit': candidate_audit}

def _channel_aware_astar_prefixed_fields(summary: dict[str, Any]) -> dict[str, Any]:
    path_cost = summary.get('path_cost_delta') if isinstance(summary.get('path_cost_delta'), dict) else {}
    channel_cost = summary.get('channel_cost_delta') if isinstance(summary.get('channel_cost_delta'), dict) else {}
    high_cost = summary.get('high_cost_exposure_delta') if isinstance(summary.get('high_cost_exposure_delta'), dict) else {}
    return {'channel_aware_astar_report_count': _int_value(summary.get('report_count')), 'channel_aware_astar_selected_count': _int_value(summary.get('selected_count')), 'channel_aware_astar_fallback_count': _int_value(summary.get('fallback_count')), 'channel_aware_astar_requested_backend_counts': dict(summary.get('requested_backend_counts', {})), 'channel_aware_astar_selected_backend_counts': dict(summary.get('selected_backend_counts', {})), 'channel_aware_astar_status_counts': dict(summary.get('status_counts', {})), 'channel_aware_astar_fallback_reason_counts': dict(summary.get('fallback_reason_counts', {})), 'channel_aware_astar_blocker_class_counts': dict(summary.get('blocker_class_counts', {})), 'channel_aware_astar_platform_goal_feasibility_class_counts': dict(summary.get('platform_goal_feasibility_class_counts', {})), 'channel_aware_astar_platform_goal_contract_mismatch_count': _int_value(summary.get('platform_goal_contract_mismatch_count')), 'channel_aware_astar_platform_goal_anchor_available_count': _int_value(summary.get('platform_goal_anchor_available_count')), 'channel_aware_astar_platform_goal_unresolved_count': _int_value(summary.get('platform_goal_unresolved_count')), 'channel_aware_astar_path_changed_count': _int_value(summary.get('path_changed_count')), 'channel_aware_astar_path_changed_rate': float(summary.get('path_changed_rate') or 0.0), 'channel_aware_astar_path_cost_delta_count': _int_value(path_cost.get('count')), 'channel_aware_astar_path_cost_delta_min': path_cost.get('min'), 'channel_aware_astar_path_cost_delta_max': path_cost.get('max'), 'channel_aware_astar_path_cost_delta_mean': path_cost.get('mean'), 'channel_aware_astar_channel_cost_delta_count': _int_value(channel_cost.get('count')), 'channel_aware_astar_channel_cost_delta_min': channel_cost.get('min'), 'channel_aware_astar_channel_cost_delta_max': channel_cost.get('max'), 'channel_aware_astar_channel_cost_delta_mean': channel_cost.get('mean'), 'channel_aware_astar_high_cost_exposure_delta_count': _int_value(high_cost.get('count')), 'channel_aware_astar_high_cost_exposure_delta_min': high_cost.get('min'), 'channel_aware_astar_high_cost_exposure_delta_max': high_cost.get('max'), 'channel_aware_astar_high_cost_exposure_delta_mean': high_cost.get('mean'), 'channel_aware_astar_candidate_audit': list(summary.get('candidate_audit', []))}

def _sampled_region_path_diagnostics(evaluations) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    start_classification_counts: Counter[str] = Counter()
    goal_classification_counts: Counter[str] = Counter()
    connector_strategy_counts: Counter[str] = Counter()
    bridge_aware_connector_status_counts: Counter[str] = Counter()
    bridge_aware_fallback_reasons: Counter[str] = Counter()
    bridge_corridor_status_counts: Counter[str] = Counter()
    bridge_corridor_fallback_reasons: Counter[str] = Counter()
    bridge_corridor_radius_counts: Counter[str] = Counter()
    reachable_component_status_counts: Counter[str] = Counter()
    reachable_component_reason_counts: Counter[str] = Counter()
    anchor_closure_status_counts: Counter[str] = Counter()
    anchor_closure_reason_counts: Counter[str] = Counter()
    anchor_closure_connection_kind_counts: Counter[str] = Counter()
    terminal_adjustment_status_counts: Counter[str] = Counter()
    terminal_adjustment_reason_counts: Counter[str] = Counter()
    execution_tie_break_status_counts: Counter[str] = Counter()
    execution_tie_break_reason_counts: Counter[str] = Counter()
    complexity_reason_counts: Counter[str] = Counter()
    selected_count = 0
    fallback_count = 0
    sample_attempt_count = 0
    candidate_ranking_count = 0
    anchor_region_added_count = 0
    anchor_region_connected_count = 0
    anchor_closure_attempt_count = 0
    anchor_closure_connected_count = 0
    connector_attempt_count = 0
    bridge_aware_connector_attempt_count = 0
    bridge_aware_connector_available_count = 0
    bridge_aware_connector_selected_count = 0
    bridge_aware_connector_rejected_count = 0
    bridge_aware_bridge_cell_count = 0
    bridge_aware_mask_added_cell_count = 0
    bridge_corridor_connector_attempt_count = 0
    bridge_corridor_connector_available_count = 0
    bridge_corridor_connector_selected_count = 0
    bridge_corridor_connector_rejected_count = 0
    bridge_corridor_added_cell_count = 0
    terminal_adjusted_count = 0
    terminal_adjustment_candidate_count = 0
    reachable_component_disconnected_count = 0
    reachable_component_replacement_selected_count = 0
    reachable_component_terminal_candidate_count = 0
    reachable_terminal_rescue_count = 0
    proxy_goal_anchor_selected_count = 0
    goal_rescue_candidate_count = 0
    benefit_surface_present_count = 0
    path_duplicate_with_baseline_count = 0
    baseline_equivalent_count = 0
    no_quality_gain_count = 0
    fixture_no_benefit_surface_count = 0
    candidate_missing_metrics_count = 0
    constrained_connector_failed_count = 0
    for item in evaluations:
        candidate = item.to_dict()
        planning_backend = candidate.get('planning_backend')
        if not isinstance(planning_backend, dict):
            continue
        sampled = planning_backend.get('sampled_region_path')
        if not isinstance(sampled, dict) or not sampled:
            continue
        status = str(sampled.get('status') or planning_backend.get('status') or 'unknown')
        status_counts[status] += 1
        if status == 'selected' or planning_backend.get('selected_backend') == 'sampled_region_path':
            selected_count += 1
        if status == 'fallback' or sampled.get('fallback_reason'):
            fallback_count += 1
        reason = sampled.get('fallback_reason')
        if reason:
            fallback_reasons[str(reason)] += 1
        constrained_connector_failed_seen = reason == 'constrained_connector_failed'
        comparison = sampled.get('candidate_comparison')
        comparison = comparison if isinstance(comparison, dict) else {}
        complexity_reason = comparison.get('complexity_reason')
        if not complexity_reason and reason in {'candidate_missing_metrics', 'constrained_connector_failed', 'fixture_no_benefit_surface', 'sampled_candidate_baseline_equivalent', 'sampled_candidate_no_quality_gain', 'sampled_candidate_path_duplicate'}:
            complexity_reason = reason
        if complexity_reason:
            complexity_reason_value = str(complexity_reason)
            complexity_reason_counts[complexity_reason_value] += 1
            if complexity_reason_value == 'sampled_candidate_baseline_equivalent':
                baseline_equivalent_count += 1
            if complexity_reason_value == 'sampled_candidate_path_duplicate':
                path_duplicate_with_baseline_count += 1
            if complexity_reason_value == 'sampled_candidate_no_quality_gain':
                no_quality_gain_count += 1
            if complexity_reason_value == 'fixture_no_benefit_surface':
                fixture_no_benefit_surface_count += 1
            if complexity_reason_value == 'candidate_missing_metrics':
                candidate_missing_metrics_count += 1
            if complexity_reason_value == 'constrained_connector_failed':
                constrained_connector_failed_seen = True
        if comparison.get('benefit_surface_present') is True:
            benefit_surface_present_count += 1
        if comparison.get('path_duplicate_with_baseline') is True and complexity_reason != 'sampled_candidate_path_duplicate':
            path_duplicate_with_baseline_count += 1
        sample_attempt_count += _int_value(sampled.get('sample_attempt_count'))
        anchoring = sampled.get('start_goal_anchoring')
        anchoring = anchoring if isinstance(anchoring, dict) else {}
        start_classification = anchoring.get('start_classification')
        goal_classification = anchoring.get('goal_classification')
        if start_classification:
            start_classification_counts[str(start_classification)] += 1
        if goal_classification:
            goal_classification_counts[str(goal_classification)] += 1
        for endpoint in ('start', 'goal'):
            if anchoring.get(f'{endpoint}_anchor_region_added') is True:
                anchor_region_added_count += 1
            if anchoring.get(f'{endpoint}_anchor_region_connected') is True:
                anchor_region_connected_count += 1
        closure = anchoring.get('anchor_connectivity_closure')
        closure = closure if isinstance(closure, dict) else {}
        anchor_closure_attempt_count += _int_value(closure.get('attempt_count'))
        anchor_closure_connected_count += _int_value(closure.get('connected_count'))
        closure_status_counts = closure.get('status_counts')
        if isinstance(closure_status_counts, dict):
            anchor_closure_status_counts.update({str(key): _int_value(value) for key, value in closure_status_counts.items()})
        closure_reason_counts = closure.get('reason_counts')
        if isinstance(closure_reason_counts, dict):
            anchor_closure_reason_counts.update({str(key): _int_value(value) for key, value in closure_reason_counts.items()})
        closure_kind_counts = closure.get('connection_kind_counts')
        if isinstance(closure_kind_counts, dict):
            anchor_closure_connection_kind_counts.update({str(key): _int_value(value) for key, value in closure_kind_counts.items()})
        sample_attempts = sampled.get('sample_attempts')
        bridge_aware_report_fallback_reasons: set[str] = set()
        bridge_corridor_report_fallback_reasons: set[str] = set()
        if isinstance(sample_attempts, list):
            for attempt in sample_attempts:
                if not isinstance(attempt, dict):
                    continue
                if attempt.get('kind') != 'connector_attempt':
                    continue
                connector_attempt_count += 1
                strategy = attempt.get('strategy')
                if strategy:
                    connector_strategy_counts[str(strategy)] += 1
                if strategy == 'bridge_aware_constrained_astar':
                    bridge_aware_connector_attempt_count += 1
                    status_value = str(attempt.get('status') or 'unknown')
                    bridge_aware_connector_status_counts[status_value] += 1
                    if status_value == 'available':
                        bridge_aware_connector_available_count += 1
                    bridge_aware_bridge_cell_count += _int_value(attempt.get('bridge_cell_count'))
                    bridge_aware_mask_added_cell_count += _int_value(attempt.get('bridge_mask_added_cell_count'))
                    bridge_reason = attempt.get('fallback_reason')
                    if bridge_reason:
                        bridge_aware_report_fallback_reasons.add(str(bridge_reason))
                if strategy == 'bridge_corridor_constrained_astar':
                    bridge_corridor_connector_attempt_count += 1
                    status_value = str(attempt.get('status') or 'unknown')
                    bridge_corridor_status_counts[status_value] += 1
                    if status_value == 'available':
                        bridge_corridor_connector_available_count += 1
                    bridge_corridor_added_cell_count += _int_value(attempt.get('bridge_corridor_added_cell_count'))
                    radius = attempt.get('bridge_corridor_radius_cells')
                    if radius is not None:
                        bridge_corridor_radius_counts[str(radius)] += 1
                    corridor_reason = attempt.get('fallback_reason') or attempt.get('bridge_corridor_failure_reason')
                    if corridor_reason:
                        bridge_corridor_report_fallback_reasons.add(str(corridor_reason))
        rankings = sampled.get('candidate_rankings')
        if isinstance(rankings, list):
            candidate_ranking_count += len(rankings)
            for ranking in rankings:
                if not isinstance(ranking, dict):
                    continue
                strategy = ranking.get('strategy')
                ranking_status = str(ranking.get('status') or 'unknown')
                if strategy == 'bridge_corridor_constrained_astar':
                    if ranking_status == 'selected':
                        bridge_corridor_connector_selected_count += 1
                    if ranking_status == 'rejected':
                        bridge_corridor_connector_rejected_count += 1
                    corridor_reason = ranking.get('fallback_reason') or ranking.get('bridge_corridor_failure_reason')
                    if corridor_reason:
                        bridge_corridor_report_fallback_reasons.add(str(corridor_reason))
                    if corridor_reason == 'constrained_connector_failed':
                        constrained_connector_failed_seen = True
                    continue
                if strategy != 'bridge_aware_constrained_astar':
                    if ranking.get('fallback_reason') == 'constrained_connector_failed':
                        constrained_connector_failed_seen = True
                    continue
                if ranking_status == 'selected':
                    bridge_aware_connector_selected_count += 1
                if ranking_status == 'rejected':
                    bridge_aware_connector_rejected_count += 1
                bridge_reason = ranking.get('fallback_reason')
                if bridge_reason:
                    bridge_aware_report_fallback_reasons.add(str(bridge_reason))
                if bridge_reason == 'constrained_connector_failed':
                    constrained_connector_failed_seen = True
        if constrained_connector_failed_seen:
            constrained_connector_failed_count += 1
        bridge_aware_fallback_reasons.update(bridge_aware_report_fallback_reasons)
        bridge_corridor_fallback_reasons.update(bridge_corridor_report_fallback_reasons)
        terminal = sampled.get('terminal_adjustment_report')
        terminal = terminal if isinstance(terminal, dict) else {}
        terminal_status = terminal.get('status')
        if terminal_status:
            terminal_adjustment_status_counts[str(terminal_status)] += 1
        terminal_reason = terminal.get('reason_code') or terminal.get('reason')
        if terminal_reason:
            terminal_adjustment_reason_counts[str(terminal_reason)] += 1
        if terminal.get('target_adjusted') is True:
            terminal_adjusted_count += 1
        terminal_adjustment_candidate_count += _int_value(terminal.get('candidate_count'))
        reachable_component_terminal_candidate_count += _int_value(terminal.get('reachable_candidate_count'))
        goal_rescue_candidate_count += _int_value(terminal.get('rescue_candidate_count'))
        if terminal.get('reachable_terminal_rescue_used') is True or terminal_reason == 'reachable_terminal_selected_by_component_projection':
            reachable_terminal_rescue_count += 1
        if terminal.get('proxy_goal_anchor_selected') is True or terminal_reason == 'proxy_goal_anchor_selected':
            proxy_goal_anchor_selected_count += 1
        component = terminal.get('reachable_component_report')
        if not isinstance(component, dict):
            component = anchoring.get('reachable_component_report')
        component = component if isinstance(component, dict) else {}
        component_status = component.get('status')
        if component_status:
            reachable_component_status_counts[str(component_status)] += 1
        component_reason = component.get('reason')
        if component_reason:
            reachable_component_reason_counts[str(component_reason)] += 1
        if component_status == 'disconnected' or component_reason == 'target_component_disconnected':
            reachable_component_disconnected_count += 1
        if terminal.get('reachable_component_replacement_selected') is True or component_reason == 'reachable_component_replacement_selected':
            reachable_component_replacement_selected_count += 1
        tie_break = sampled.get('execution_tie_break')
        tie_break = tie_break if isinstance(tie_break, dict) else {}
        tie_break_status = tie_break.get('status')
        if tie_break_status:
            execution_tie_break_status_counts[str(tie_break_status)] += 1
        tie_break_reason = tie_break.get('reason')
        if tie_break_reason:
            execution_tie_break_reason_counts[str(tie_break_reason)] += 1
        graph = candidate.get('region_graph')
        if isinstance(graph, dict):
            source_counts[str(graph.get('graph_source') or graph.get('region_source') or 'unknown')] += 1
        else:
            source_counts['unknown'] += 1
    return {
        "selected_count": selected_count,
        "fallback_count": fallback_count,
        "status_counts": dict(sorted(status_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "sample_attempt_count": sample_attempt_count,
        "candidate_ranking_count": candidate_ranking_count,
        "anchor_region_added_count": anchor_region_added_count,
        "anchor_region_connected_count": anchor_region_connected_count,
        "anchor_closure_attempt_count": anchor_closure_attempt_count,
        "anchor_closure_connected_count": anchor_closure_connected_count,
        "anchor_closure_status_counts": dict(sorted(anchor_closure_status_counts.items())),
        "anchor_closure_reason_counts": dict(sorted(anchor_closure_reason_counts.items())),
        "anchor_closure_connection_kind_counts": dict(sorted(anchor_closure_connection_kind_counts.items())),
        "start_classification_counts": dict(sorted(start_classification_counts.items())),
        "goal_classification_counts": dict(sorted(goal_classification_counts.items())),
        "connector_attempt_count": connector_attempt_count,
        "connector_strategy_counts": dict(sorted(connector_strategy_counts.items())),
        "bridge_aware_connector_attempt_count": bridge_aware_connector_attempt_count,
        "bridge_aware_connector_available_count": bridge_aware_connector_available_count,
        "bridge_aware_connector_selected_count": bridge_aware_connector_selected_count,
        "bridge_aware_connector_rejected_count": bridge_aware_connector_rejected_count,
        "bridge_aware_connector_status_counts": dict(sorted(bridge_aware_connector_status_counts.items())),
        "bridge_aware_fallback_reasons": dict(sorted(bridge_aware_fallback_reasons.items())),
        "bridge_aware_bridge_cell_count": bridge_aware_bridge_cell_count,
        "bridge_aware_mask_added_cell_count": bridge_aware_mask_added_cell_count,
        "bridge_corridor_connector_attempt_count": bridge_corridor_connector_attempt_count,
        "bridge_corridor_connector_available_count": bridge_corridor_connector_available_count,
        "bridge_corridor_connector_selected_count": bridge_corridor_connector_selected_count,
        "bridge_corridor_connector_rejected_count": bridge_corridor_connector_rejected_count,
        "bridge_corridor_status_counts": dict(sorted(bridge_corridor_status_counts.items())),
        "bridge_corridor_fallback_reasons": dict(sorted(bridge_corridor_fallback_reasons.items())),
        "bridge_corridor_radius_counts": dict(sorted(bridge_corridor_radius_counts.items())),
        "bridge_corridor_added_cell_count": bridge_corridor_added_cell_count,
        "terminal_adjusted_count": terminal_adjusted_count,
        "terminal_adjustment_candidate_count": terminal_adjustment_candidate_count,
        "terminal_adjustment_status_counts": dict(sorted(terminal_adjustment_status_counts.items())),
        "terminal_adjustment_reason_counts": dict(sorted(terminal_adjustment_reason_counts.items())),
        "reachable_component_status_counts": dict(sorted(reachable_component_status_counts.items())),
        "reachable_component_reason_counts": dict(sorted(reachable_component_reason_counts.items())),
        "reachable_component_disconnected_count": reachable_component_disconnected_count,
        "reachable_component_replacement_selected_count": reachable_component_replacement_selected_count,
        "reachable_component_terminal_candidate_count": reachable_component_terminal_candidate_count,
        "reachable_terminal_rescue_count": reachable_terminal_rescue_count,
        "proxy_goal_anchor_selected_count": proxy_goal_anchor_selected_count,
        "goal_rescue_candidate_count": goal_rescue_candidate_count,
        "benefit_surface_present_count": benefit_surface_present_count,
        "path_duplicate_with_baseline_count": path_duplicate_with_baseline_count,
        "baseline_equivalent_count": baseline_equivalent_count,
        "no_quality_gain_count": no_quality_gain_count,
        "fixture_no_benefit_surface_count": fixture_no_benefit_surface_count,
        "candidate_missing_metrics_count": candidate_missing_metrics_count,
        "constrained_connector_failed_count": constrained_connector_failed_count,
        "complexity_reason_counts": dict(sorted(complexity_reason_counts.items())),
        "execution_tie_break_status_counts": dict(sorted(execution_tie_break_status_counts.items())),
        "execution_tie_break_reason_counts": dict(sorted(execution_tie_break_reason_counts.items())),
    }
__all__ = ('_iris_diagnostics', '_region_graph_diagnostics', '_convex_region_diagnostics', '_gcs_trajectory_diagnostics', '_gcs_candidate_diagnostics', '_gcs_control_point_diagnostics', '_gcs_motion_feasibility_diagnostics', '_gcs_curvature_constrained_diagnostics', '_channel_aware_astar_diagnostics', '_channel_aware_astar_blocker_class', '_channel_aware_astar_failure_taxonomy', '_platform_goal_failure_class', '_platform_goal_contract_mismatch', '_aggregate_channel_aware_astar_diagnostics', '_channel_aware_astar_prefixed_fields', '_sampled_region_path_diagnostics')
