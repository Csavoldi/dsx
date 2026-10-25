//! Explicit tie-breaking for adaptive expert residency. Score policy is unchanged.
pub type Entry = (f32, usize, usize);

pub fn candidates(entries: &mut [Entry], stable: bool) {
    entries.sort_unstable_by(|a,b| {
        let score=b.0.total_cmp(&a.0);
        if stable {score.then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))} else {score}
    });
}

/// Coldest last for pop(); ties evict the smallest (layer, expert) first.
pub fn victims(entries: &mut [Entry], stable: bool) {
    entries.sort_unstable_by(|a,b| {
        let score=b.0.total_cmp(&a.0);
        if stable {score.then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2))} else {score}
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_scores_choose_the_same_victim_for_every_input_order() {
        let values=[(1.0,6,358),(1.0,17,335),(1.0,25,9)];
        let permutations=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
        let mut legacy=std::collections::BTreeSet::new();
        for order in permutations {
            let mut entries=order.map(|i|values[i]);victims(&mut entries,false);legacy.insert((entries[2].1,entries[2].2));
            let mut entries=order.map(|i|values[i]);victims(&mut entries,true);assert_eq!(entries[2],values[0]);
        }
        assert!(legacy.len()>1,"Score-only eviction must expose the order-dependent baseline");
    }
    #[test]
    fn distinct_scores_keep_the_existing_policy() {
        let source=[(2.0,6,358),(1.0,17,335),(3.0,25,9)];
        for stable in [false,true] {
            let mut entries=source;candidates(&mut entries,stable);assert_eq!(entries[0],source[2]);
            let mut entries=source;victims(&mut entries,stable);assert_eq!(entries[2],source[1]);
        }
    }
}
