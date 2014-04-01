/* ===========================================================================

	Project: VoxRender

	Description: Implements a control widget for editing the light set

    Copyright (C) 2014 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Include Headers
#include "lightingwidget.h"

#include <QLayout>

// Include Dependencies
#include "VoxLib/Action/ActionManager.h"
#include "Actions/AddRemLightAct.h"
#include "mainwindow.h"
#include "pointlightwidget.h"
#include "ambientlightwidget.h"

using namespace vox;

// --------------------------------------------------------------------
//  Constructor
// --------------------------------------------------------------------
LightingWidget::LightingWidget(QWidget * parent, QLayout * layout) : 
    QWidget(parent),
    m_layout(layout),
    m_parent(parent)
{
    m_parent = parent;

    connect(MainWindow::instance, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));
    
	// Set alignment of panes within lighting tab layout 
	m_layout->setAlignment(Qt::AlignTop);
    m_spacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding );

    // Create new pane for the ambient light setting widget
    m_ambientPane = new PaneWidget(m_parent);
    QWidget * currWidget = new AmbientLightWidget(m_ambientPane); 

    m_ambientPane->setTitle("Environment");
    m_ambientPane->setIcon(":/icons/lightgroupsicon.png");
    m_ambientPane->setWidget(currWidget);
    m_ambientPane->expand();

    m_layout->addWidget(m_ambientPane);

    // Reinsert spacer following new pane
	m_layout->addItem(m_spacer);
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
LightingWidget::~LightingWidget()
{
    m_layout->removeItem(m_spacer);
    delete m_spacer;
}

// ----------------------------------------------------------------------------
//  Adds a control widget for an existing light object 
// ----------------------------------------------------------------------------
void LightingWidget::add(std::shared_ptr<Light> light)
{
    // Remove spacer prior to pane insertion
    m_layout->removeItem(m_spacer);

    // Create new pane for the light setting widget
    PaneWidget * pane = new PaneWidget(m_parent);
   
    QWidget * currWidget = new PointLightWidget(pane, light); 

    pane->showOnOffButton();
    pane->showVisibilityButtons();
    pane->setTitle("Point Light");
    pane->setIcon(":/icons/lightgroupsicon.png");
    pane->setWidget(currWidget);
    pane->expand();
    pane->setOn(light->isVisible());

    connect(pane, SIGNAL(removed(PaneWidget *)), this, SLOT(remove(PaneWidget *)));

    m_layout->addWidget(pane);

    m_lightPanes.push_back(pane);

    // Reinsert spacer following new pane
	m_layout->addItem(m_spacer);

    // Register the visibility change event callback
    light->onVisibilityChanged([=] (bool isVisible, bool suppress) {
        auto functor = [=] () { light->setVisible(!light->isVisible(), true); light->setDirty(); };
        if (!suppress) ActionManager::instance().push(functor, functor);
        pane->setOn(isVisible);
    });
}

// ----------------------------------------------------------------------------
//  Removes a light from the active scene
// ----------------------------------------------------------------------------
void LightingWidget::remove(PaneWidget * pane)
{
    m_lightPanes.remove(pane);

    m_layout->removeWidget(pane);
    delete pane;
}

// ----------------------------------------------------------------------------
//  Removes a light from the active scene
// ----------------------------------------------------------------------------
void LightingWidget::remove(std::shared_ptr<Light> light)
{
    BOOST_FOREACH (auto pane, m_lightPanes)
    {
        auto ptr = dynamic_cast<PointLightWidget*>(pane->getWidget());
        if (ptr && ptr->light().get() == light.get())
        {
            m_lightPanes.remove(pane);

            m_layout->removeWidget(pane);
            delete pane;

            return;
        }
    }
}

// --------------------------------------------------------------------
//  Slot for handling scene change events
// --------------------------------------------------------------------
void LightingWidget::sceneChanged()
{
    auto scene = MainWindow::instance->scene();
    
    // Synchronize the lighting controls
    BOOST_FOREACH (auto & pane, m_lightPanes) delete pane;
    m_lightPanes.clear();
    BOOST_FOREACH (auto & light, scene.lightSet->lights()) 
        add(light);

    // Connect to the light callback events for event detection
    scene.lightSet->onAdd([this] (std::shared_ptr<Light> light, bool suppress) {
        if (!suppress) ActionManager::instance().push(AddRemLightAct::create(light));
        add(light); });
    scene.lightSet->onRemove([this] (std::shared_ptr<Light> light, bool suppress) {
        if (!suppress) ActionManager::instance().push(AddRemLightAct::create(light));
        remove(light); });
}