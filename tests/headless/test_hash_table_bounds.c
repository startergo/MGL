/* hash_table.m insertHashElement: a name that cannot fit after the growth loop
 * reaches its UINT_MAX-entry cap (name == UINT_MAX: valid slots are 0..UINT_MAX-1)
 * must be refused, not written one slot past the end of the table.  The table here
 * is already at the cap but backed by a tiny buffer, so an unchecked write lands
 * ~64 GB past it and faults; the normal growth path must still work. */
#include "mgl_test.h"
#include "hash_table.h"
#include <limits.h>
#include <stdlib.h>

TEST(name_uint_max_refused)
{
    HashObj *small = calloc(4, sizeof(HashObj));
    HashTable t = { UINT_MAX, 0, small };
    int marker = 1;

    insertHashElement(&t, UINT_MAX, &marker);          /* must not write */
    EXPECT_TRUE(searchHashTable(&t, UINT_MAX) == NULL);
    EXPECT_TRUE(small[0].data == NULL && small[3].data == NULL);
    free(small);
}

TEST(growth_still_inserts)
{
    HashTable t;
    int a = 1, b = 2;

    initHashTable(&t, 4);
    insertHashElement(&t, 2, &a);
    insertHashElement(&t, 100, &b);                    /* grows 4 -> 128 */
    EXPECT_TRUE(t.size > 100);
    EXPECT_TRUE(searchHashTable(&t, 2) == &a);
    EXPECT_TRUE(searchHashTable(&t, 100) == &b);
    EXPECT_TRUE(searchHashTable(&t, 99) == NULL);
}

int main(void)
{
    RUN(name_uint_max_refused);
    RUN(growth_still_inserts);
    return mgl_test_summary();
}
