{#
  This custom template is a bridge between sphinx-autoapi and autodoc-pydantic.
  By using '.. autoclass::', we delegate the rendering to sphinx.ext.autodoc,
  allowing autodoc-pydantic to correctly process Pydantic models (Field descriptions, etc.),
  which is not possible with the default sphinx-autoapi static analysis.
#}
{% if obj.display %}
.. autoclass:: {{ obj.id }}
   :members:
   :undoc-members:
   :show-inheritance:
{% endif %}
