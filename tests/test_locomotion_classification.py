"""
Unit tests for locomotion classification in dataio.load_fictrac_data.

These tests focus on the classification_method parameter behavior
without requiring actual Fictrac data or ImagingDataObject.
"""
import numpy as np
import pytest


def classify_locomotion_state(trial_mean_values, classification_method):
    """
    Standalone classification logic extracted for testing.
    
    This mirrors the classification logic in dataio.load_fictrac_data.
    
    Args:
        trial_mean_values: 1D or 2D array of trial mean values (e.g., walking magnitude)
        classification_method: 'auto', dict with bounds, or callable
        
    Returns:
        locomotion_state: int array with 0=stationary, 1=locomoting, -1=ambiguous
        is_locomoting: bool array (True=locomoting)
    """
    trial_mean_values = np.atleast_1d(trial_mean_values)
    
    if classification_method == 'auto':
        # Simulate auto behavior with a simple threshold
        from skimage import filters
        thresh = filters.threshold_li(trial_mean_values.flatten())
        is_locomoting = trial_mean_values > thresh
        locomotion_state = is_locomoting.astype('int')
    elif isinstance(classification_method, dict):
        stationary_upper = classification_method.get('stationary_upper')
        locomoting_lower = classification_method.get('locomoting_lower')
        
        if stationary_upper is None or locomoting_lower is None:
            raise ValueError("classification_method dict must have 'stationary_upper' and 'locomoting_lower' keys")
        
        locomotion_state = np.full(trial_mean_values.shape, -1, dtype=int)
        locomotion_state[trial_mean_values <= stationary_upper] = 0
        locomotion_state[trial_mean_values >= locomoting_lower] = 1
        is_locomoting = locomotion_state == 1
    elif callable(classification_method):
        result = classification_method(trial_mean_values)
        if isinstance(result, dict):
            locomotion_state = np.full(trial_mean_values.shape, -1, dtype=int)
            locomotion_state[result.get('stationary', np.zeros_like(trial_mean_values, dtype=bool))] = 0
            locomotion_state[result.get('locomoting', np.zeros_like(trial_mean_values, dtype=bool))] = 1
        else:
            locomotion_state = np.asarray(result, dtype=int)
        is_locomoting = locomotion_state == 1
    else:
        raise ValueError(f'Unrecognized classification_method: {classification_method}')
    
    return locomotion_state, is_locomoting


class TestClassificationMethodDict:
    """Tests for dict-based two-threshold classification."""
    
    def test_basic_classification(self):
        """Test that values are correctly classified using upper/lower bounds."""
        trial_values = np.array([5, 15, 30, 60, 100])
        method = {'stationary_upper': 20, 'locomoting_lower': 50}
        
        state, is_locomoting = classify_locomotion_state(trial_values, method)
        
        # Stationary: <= 20
        assert state[0] == 0  # 5
        assert state[1] == 0  # 15
        # Ambiguous: > 20 and < 50
        assert state[2] == -1  # 30
        # Locomoting: >= 50
        assert state[3] == 1  # 60
        assert state[4] == 1  # 100
        
    def test_is_locomoting_matches_locomoting(self):
        """Test that is_locomoting is True only for locomoting trials."""
        trial_values = np.array([10, 25, 80])
        method = {'stationary_upper': 20, 'locomoting_lower': 50}
        
        state, is_locomoting = classify_locomotion_state(trial_values, method)
        
        np.testing.assert_array_equal(is_locomoting, [False, False, True])
        
    def test_edge_cases_at_boundaries(self):
        """Test that boundary values are inclusive."""
        trial_values = np.array([20, 20.001, 49.999, 50])
        method = {'stationary_upper': 20, 'locomoting_lower': 50}
        
        state, _ = classify_locomotion_state(trial_values, method)
        
        assert state[0] == 0   # exactly at stationary_upper -> stationary
        assert state[1] == -1  # just above -> ambiguous
        assert state[2] == -1  # just below locomoting_lower -> ambiguous
        assert state[3] == 1   # exactly at locomoting_lower -> locomoting
        
    def test_overlapping_thresholds(self):
        """Test behavior when stationary_upper >= locomoting_lower (no ambiguous zone)."""
        trial_values = np.array([10, 30, 50])
        method = {'stationary_upper': 30, 'locomoting_lower': 30}
        
        state, _ = classify_locomotion_state(trial_values, method)
        
        # 10 <= 30 -> stationary
        assert state[0] == 0
        # 30 <= 30 AND 30 >= 30, so it gets locomoting (last assignment wins)
        assert state[1] == 1
        # 50 >= 30 -> locomoting
        assert state[2] == 1
        
    def test_missing_keys_raises_error(self):
        """Test that missing keys raise ValueError."""
        trial_values = np.array([10, 20, 30])
        
        with pytest.raises(ValueError, match="stationary_upper"):
            classify_locomotion_state(trial_values, {'locomoting_lower': 50})
            
        with pytest.raises(ValueError, match="locomoting_lower"):
            classify_locomotion_state(trial_values, {'stationary_upper': 20})


class TestClassificationMethodCallable:
    """Tests for callable classification method."""
    
    def test_callable_returning_int_array(self):
        """Test callable that returns int array directly."""
        trial_values = np.array([10, 30, 80])
        
        def my_classifier(values):
            result = np.full_like(values, -1, dtype=int)
            result[values < 20] = 0
            result[values > 60] = 1
            return result
        
        state, is_locomoting = classify_locomotion_state(trial_values, my_classifier)
        
        np.testing.assert_array_equal(state, [0, -1, 1])
        np.testing.assert_array_equal(is_locomoting, [False, False, True])
        
    def test_callable_returning_dict(self):
        """Test callable that returns dict with bool arrays."""
        trial_values = np.array([10, 30, 80, 90])
        
        def my_classifier(values):
            return {
                'stationary': values < 20,
                'locomoting': values > 60
            }
        
        state, is_locomoting = classify_locomotion_state(trial_values, my_classifier)
        
        np.testing.assert_array_equal(state, [0, -1, 1, 1])
        np.testing.assert_array_equal(is_locomoting, [False, False, True, True])
        
    def test_callable_with_custom_logic(self):
        """Test callable with complex custom logic."""
        trial_values = np.array([5, 15, 25, 35, 45])
        
        # Custom logic: stationary if < 15, walking if > 30 AND odd index
        def complex_classifier(values):
            state = np.full_like(values, -1, dtype=int)
            state[values < 15] = 0
            for i, v in enumerate(values):
                if v > 30 and i % 2 == 1:
                    state[i] = 1
            return state
        
        state, _ = classify_locomotion_state(trial_values, complex_classifier)
        
        # values = [5, 15, 25, 35, 45]
        # values < 15 = [True, False, False, False, False]
        # So state starts as [0, -1, -1, -1, -1]
        # After loop: 35 > 30 and index 3 is odd -> state[3] = 1
        # 45 > 30 and index 4 is even -> stays -1
        np.testing.assert_array_equal(state, [0, -1, -1, 1, -1])


class TestClassificationMethodAuto:
    """Tests for 'auto' classification method."""
    
    def test_auto_uses_threshold(self):
        """Test that auto mode produces binary classification."""
        # Create bimodal distribution
        trial_values = np.array([5, 10, 8, 100, 120, 90, 5, 110])
        
        state, is_locomoting = classify_locomotion_state(trial_values, 'auto')
        
        # Should only produce 0 and 1 (no -1 for auto mode)
        assert set(np.unique(state)).issubset({0, 1})
        # High values should be locomoting
        assert np.all(state[trial_values > 50] == 1)
        # Low values should be stationary
        assert np.all(state[trial_values < 20] == 0)


class TestClassificationMethodInvalid:
    """Tests for invalid classification methods."""
    
    def test_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError."""
        trial_values = np.array([10, 20, 30])
        
        with pytest.raises(ValueError, match="Unrecognized"):
            classify_locomotion_state(trial_values, 'invalid_method')
            
    def test_invalid_type_raises_error(self):
        """Test that invalid types raise ValueError."""
        trial_values = np.array([10, 20, 30])
        
        with pytest.raises(ValueError):
            classify_locomotion_state(trial_values, 123)
            
        with pytest.raises(ValueError):
            classify_locomotion_state(trial_values, [20, 50])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
