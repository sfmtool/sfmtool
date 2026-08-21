use super::*;

#[test]
fn test_gather_rows_reorders_whole_rows() {
    let rows = [0.0, 0.1, 1.0, 1.1, 2.0, 2.1];
    let out = gather_rows(&rows, 2, [2usize, 0, 1]);
    assert_eq!(out, vec![2.0, 2.1, 0.0, 0.1, 1.0, 1.1]);
}

#[test]
fn test_gather_rows_empty_selection_is_empty() {
    let rows = [1u8, 2, 3, 4];
    let out = gather_rows(&rows, 2, std::iter::empty());
    assert!(out.is_empty());
}

#[test]
fn test_gather_rows_may_repeat_and_skip_rows() {
    let rows = [10u8, 11, 20, 21, 30, 31];
    let out = gather_rows(&rows, 2, [1usize, 1]);
    assert_eq!(out, vec![20, 21, 20, 21]);
}

/// The two-level form the polar sweep uses: sort indices into a filtered
/// subset, composed with that subset's indices into the caller's arrays.
#[test]
fn test_gather_rows_composes_two_index_levels() {
    // Rows 1 and 3 survived a filter; the sweep then ordered them [1, 0].
    let rows = [0.0, 10.0, 20.0, 30.0];
    let valid = [1usize, 3];
    let sort_idx = [1usize, 0];
    let out = gather_rows(&rows, 1, sort_idx.iter().map(|&si| valid[si]));
    assert_eq!(out, vec![30.0, 10.0]);
}

/// Every parallel array must come out in the same feature order, whatever its
/// stride — that is the invariant a window index relies on when it names a
/// descriptor, a position and an affine shape interchangeably.
#[test]
fn test_gather_rows_agrees_across_strides() {
    let order = [2usize, 0, 1];
    let positions = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
    let affines = [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3];

    let gathered_positions = gather_rows(&positions, 2, order);
    let gathered_affines = gather_rows(&affines, 4, order);

    // Whole rows, not just their first element: a bug that started each row at
    // the right offset but pulled its remaining elements from elsewhere would
    // otherwise pass.
    for (slot, &src) in order.iter().enumerate() {
        assert_eq!(
            &gathered_positions[slot * 2..slot * 2 + 2],
            &positions[src * 2..src * 2 + 2]
        );
        assert_eq!(
            &gathered_affines[slot * 4..slot * 4 + 4],
            &affines[src * 4..src * 4 + 4]
        );
    }
}
