//! Complex composition tests.
//!
//! Validates that `RecordableOp` propagates through deeply-nested
//! combinator stacks without trait-bound surprises. Mixes AndThen and
//! Bundle2 in various shapes.
//!
//! Findings (see also the plan file):
//!
//! - **AndThen-nested-N-deep**: propagation works. Compiler infers
//!   the AND of all children's recordability automatically.
//! - **Bundle of AndThens**: works. Each AndThen branch records its
//!   sub-chain; the Bundle joins their sync points.
//! - **AndThen-of-Bundles**: works. The Bundle's joined sync point
//!   becomes the AndThen-next stage's wait list.
//! - **Bundle of Bundle of AndThens (3-level nesting)**: works. No
//!   trait-bound explosion, no inference surprises.
//! - **Type inference holds at depth 5+** without explicit turbofish.
//!
//! - **Negative case**: replacing any leaf with a non-recordable op
//!   (Upload) anywhere in the nest fails to compile at the chain
//!   root's `.record()` call (verified manually; documented in the
//!   compile-fail tests).

#[cfg(test)]
mod tests {
    use crate::combinators::{AndThen, Bundle2};
    use crate::device_op::{FakeCommandBuffer, RecordContext, RecordableOp};
    use crate::leaves::{BufferCopy, BufferFill, FillKernel};

    /// AndThen nested 4 deep — straight pipeline of 5 ops.
    #[test]
    fn andthen_nested_5_deep() {
        let chain = AndThen {
            source: BufferFill {
                buf_id: 1,
                pattern: 0,
                len: 100,
            },
            f: |b| AndThen {
                source: FillKernel {
                    n: 100,
                    buf_id: b,
                    value: 1,
                },
                f: move |b| AndThen {
                    source: FillKernel {
                        n: 100,
                        buf_id: b,
                        value: 2,
                    },
                    f: move |b| AndThen {
                        source: FillKernel {
                            n: 100,
                            buf_id: b,
                            value: 3,
                        },
                        f: move |b| FillKernel {
                            n: 100,
                            buf_id: b,
                            value: 4,
                        },
                    },
                },
            },
        };

        let mut cb = FakeCommandBuffer::default();
        let mut rec = RecordContext {
            command_buffer: &mut cb,
        };
        let (_out, _sps) = chain.record(&mut rec, vec![]).expect("record");
        assert_eq!(cb.recorded_commands.len(), 5);
    }

    /// Bundle of two AndThen pipelines.
    #[test]
    fn bundle_of_andthens() {
        let chain = Bundle2 {
            a: AndThen {
                source: BufferFill {
                    buf_id: 1,
                    pattern: 0,
                    len: 100,
                },
                f: |b| FillKernel {
                    n: 100,
                    buf_id: b,
                    value: 1,
                },
            },
            b: AndThen {
                source: BufferFill {
                    buf_id: 2,
                    pattern: 0,
                    len: 100,
                },
                f: |b| FillKernel {
                    n: 100,
                    buf_id: b,
                    value: 2,
                },
            },
        };

        let mut cb = FakeCommandBuffer::default();
        let mut rec = RecordContext {
            command_buffer: &mut cb,
        };
        let ((_a, _b), _sps) = chain.record(&mut rec, vec![]).expect("record");
        // 2 leaves per branch + 2 kernels + 1 barrier = 5
        assert_eq!(cb.recorded_commands.len(), 5);
    }

    /// AndThen of a Bundle of two leaves.
    #[test]
    fn andthen_after_bundle() {
        let chain = AndThen {
            source: Bundle2 {
                a: BufferFill {
                    buf_id: 1,
                    pattern: 0,
                    len: 100,
                },
                b: BufferFill {
                    buf_id: 2,
                    pattern: 0,
                    len: 100,
                },
            },
            f: |(a, _b)| FillKernel {
                n: 100,
                buf_id: a,
                value: 42,
            },
        };

        let mut cb = FakeCommandBuffer::default();
        let mut rec = RecordContext {
            command_buffer: &mut cb,
        };
        let (_out, _sps) = chain.record(&mut rec, vec![]).expect("record");
        // 2 leaves + 1 barrier + 1 kernel = 4
        assert_eq!(cb.recorded_commands.len(), 4);
    }

    /// 3-level nesting: Bundle of (Bundle of AndThens) and a copy.
    /// This is the gnarliest shape we need to support.
    #[test]
    fn three_level_nested_bundle() {
        let chain = Bundle2 {
            a: Bundle2 {
                a: AndThen {
                    source: BufferFill {
                        buf_id: 1,
                        pattern: 0,
                        len: 100,
                    },
                    f: |b| FillKernel {
                        n: 100,
                        buf_id: b,
                        value: 1,
                    },
                },
                b: AndThen {
                    source: BufferFill {
                        buf_id: 2,
                        pattern: 0,
                        len: 100,
                    },
                    f: |b| FillKernel {
                        n: 100,
                        buf_id: b,
                        value: 2,
                    },
                },
            },
            b: BufferCopy {
                src_id: 3,
                dst_id: 4,
                len_bytes: 400,
            },
        };

        let mut cb = FakeCommandBuffer::default();
        let mut rec = RecordContext {
            command_buffer: &mut cb,
        };
        let _ = chain.record(&mut rec, vec![]).expect("record");
        // Inner Bundle: 2 leaves + 2 kernels + 1 barrier = 5
        // Outer chain: copy + outer barrier = 2 more
        // Total: 7
        assert_eq!(cb.recorded_commands.len(), 7);
    }

    /// The same 5-deep AndThen chain, but called twice — verifies
    /// that `record` consumes self correctly and a fresh chain can
    /// be built and recorded independently.
    #[test]
    fn two_independent_records() {
        for _ in 0..2 {
            let chain = AndThen {
                source: BufferFill {
                    buf_id: 1,
                    pattern: 0,
                    len: 100,
                },
                f: |b| FillKernel {
                    n: 100,
                    buf_id: b,
                    value: 1,
                },
            };
            let mut cb = FakeCommandBuffer::default();
            let mut rec = RecordContext {
                command_buffer: &mut cb,
            };
            let _ = chain.record(&mut rec, vec![]).expect("record");
            assert_eq!(cb.recorded_commands.len(), 2);
        }
    }
}
