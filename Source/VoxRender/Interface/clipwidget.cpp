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
#include "clipwidget.h"

#include <QLayout>

// Include Dependencies
#include "VoxLib/Action/ActionManager.h"
#include "VoxScene/PrimGroup.h"
#include "Actions/AddRemLightAct.h"
#include "mainwindow.h"
#include "clipplanewidget.h"
#include "Actions/AddRemClipAct.h"

using namespace vox;

// --------------------------------------------------------------------
//  Constructor
// --------------------------------------------------------------------
ClipWidget::ClipWidget(QWidget * parent, QLayout * layout) : 
    QWidget(parent),
    m_layout(layout)
{
    connect(MainWindow::instance, SIGNAL(sceneChanged(vox::Scene &,void *)), 
            this, SLOT(sceneChanged(vox::Scene &,void *)), Qt::DirectConnection);
    
	// Set alignment of panes within lighting tab layout 
	m_layout->setAlignment(Qt::AlignTop);
    m_spacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    // Reinsert spacer following new pane
	m_layout->addItem(m_spacer);
}

// --------------------------------------------------------------------
//  Destructor
// --------------------------------------------------------------------
ClipWidget::~ClipWidget()
{
    m_layout->removeItem(m_spacer);
    delete m_spacer;
}

// ----------------------------------------------------------------------------
//  Adds a control widget for an existing light object 
// ----------------------------------------------------------------------------
void ClipWidget::add(std::shared_ptr<Primitive> prim)
{
    // Remove spacer prior to pane insertion
    m_layout->removeItem(m_spacer);

    // Create new pane for the light setting widget
    PaneWidget *pane = new PaneWidget(parentWidget());
   
    // Create the control widget to populate the pane
    QWidget * currWidget = nullptr;
    if (prim->typeId() == Plane::classTypeId())
    {
        auto plane = std::dynamic_pointer_cast<vox::Plane>(prim);
        if (!plane) throw Error(__FILE__, __LINE__, "GUI", 
            "Error interpreting primitive :TODO:");
        currWidget = new ClipPlaneWidget(pane, this, plane); 
    }
    else
    {
        VOX_LOG_WARNING(Error_NotImplemented, "GUI",
            format("Geometry type '%1%' unrecognized. '%2%' will not be editable.", prim->typeId(), prim->idString()));

        return;
    }

    pane->showOnOffButton();
    pane->showVisibilityButtons();
    pane->setTitle("Clip Plane");
    pane->setIcon(":/icons/lightgroupsicon.png");
    pane->setWidget(currWidget);
    pane->expand();
    pane->setOn(prim->isVisible());
    
    connect(pane, SIGNAL(removed(PaneWidget *)), this, SLOT(remove(PaneWidget *)));

    m_layout->addWidget(pane);

    m_panes.push_back(pane);

    // Reinsert spacer following new pane
	m_layout->addItem(m_spacer);

    // Register the visibility change event callback
    prim->onVisibilityChanged([=] (bool isVisible, bool suppress) {
        auto functor = [=] () { prim->setVisible(!prim->isVisible(), true); prim->setDirty(); };
        if (!suppress) ActionManager::instance().push(functor, functor);
        pane->setOn(isVisible);
    });
}

// ----------------------------------------------------------------------------
//  Removes a clipping primitive from the active scene
// ----------------------------------------------------------------------------
void ClipWidget::remove(PaneWidget * pane)
{
    m_panes.remove(pane);
    
    auto clipwidget = dynamic_cast<ClipPlaneWidget*>(pane->getWidget());
    if (clipwidget) 
    {
        auto scene = MainWindow::instance->scene();
        scene->clipGeometry->remove(clipwidget->plane());
        scene->clipGeometry->setDirty();
    }

    m_layout->removeWidget(pane);
    delete pane;
}

// ----------------------------------------------------------------------------
//  Removes a clipping primitive's pane from the display
// ----------------------------------------------------------------------------
void ClipWidget::remove(std::shared_ptr<Primitive> prim)
{
    BOOST_FOREACH (auto pane, m_panes)
    {
        auto ptr = dynamic_cast<ClipPlaneWidget*>(pane->getWidget());
        if (!ptr) continue;

        if (ptr && ptr->plane() == prim)
        {
            m_panes.remove(pane);

            m_layout->removeWidget(pane);
            delete pane;

            return;
        }
    }
}

// --------------------------------------------------------------------
//  Slot for handling scene change events
// --------------------------------------------------------------------
void ClipWidget::sceneChanged(Scene & scene, void * userInfo)
{
    if (!scene.clipGeometry || userInfo == this) return;
    
    // Connect to the geometry callback events for event detection
    if (userInfo == MainWindow::instance) // initial load by mainwindow
    {
        scene.clipGeometry->onAdd([this] (std::shared_ptr<Primitive> prim, bool suppress) {
            if (!suppress) ActionManager::instance().push(AddRemClipAct::create(prim));
            add(prim); });
        scene.clipGeometry->onRemove([this] (std::shared_ptr<Primitive> prim, bool suppress) {
            if (!suppress) ActionManager::instance().push(AddRemClipAct::create(prim));
            remove(prim); });
    }

    // Synchronize the lighting controls
    while (!m_panes.empty())
    {
        auto pane = m_panes.back();
        m_panes.pop_back();
        delete pane;
    }
    BOOST_FOREACH (auto & prim, scene.clipGeometry->children()) add(prim);
}