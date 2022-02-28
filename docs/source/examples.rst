Examples
========

.. jinja:: examples_context

    {% for ex in examples %}

    .. literalinclude:: {{ex}}
        :caption: Example {{loop.index}}
    {% endfor %}
