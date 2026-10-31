//! Checks that usearch builds, searches, and deletes on the current target.

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

fn main() {
    let options = IndexOptions {
        dimensions: 3,
        metric: MetricKind::IP,
        quantization: ScalarKind::F32,
        connectivity: 16,
        expansion_add: 200,
        expansion_search: 64,
        multi: false,
    };
    let index = Index::new(&options).expect("index builds");
    index.reserve(10).expect("capacity reserved");
    index.add(1, &[1.0, 0.0, 0.0]).expect("first add");
    index.add(2, &[0.0, 1.0, 0.0]).expect("second add");

    let found = index.search(&[1.0, 0.0, 0.0], 1).expect("search");
    assert_eq!(found.keys, [1]);

    index.remove(1).expect("remove");
    let found = index.search(&[1.0, 0.0, 0.0], 1).expect("search after remove");
    assert_eq!(found.keys, [2]);
    println!("ok");
}
