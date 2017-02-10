/*******************************************************************************
 * UW SPF - The University of Washington Semantic Parsing Framework
 * <p>
 * Copyright (C) 2013 Yoav Artzi
 * <p>
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or any later version.
 * <p>
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 ******************************************************************************/
package edu.uw.cs.tiny.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import junit.framework.Assert;

import org.junit.Test;

import edu.uw.cs.lil.tiny.utils.PowerSet;
import edu.uw.cs.utils.collections.ListUtils;

public class PowerSetTest {
	
	@Test
	public void test() {
		final Integer[] nums = { 1, 2 };
		final Set<List<Integer>> powerset = new HashSet<List<Integer>>();
		powerset.add(new ArrayList<Integer>());
		powerset.add(ListUtils.createList(1));
		powerset.add(ListUtils.createList(2));
		powerset.add(ListUtils.createList(1, 2));
		for (final List<Integer> subset : new PowerSet<Integer>(
				Arrays.asList(nums))) {
			Assert.assertTrue(String.format("Failed to remove %s", subset),
					powerset.remove(new ArrayList<Integer>(subset)));
		}
		Assert.assertTrue(powerset.isEmpty());
	}
	
	@Test
	public void test2() {
		final List<Integer> list = new LinkedList<Integer>();
		for (int i = 0; i < 62; ++i) {
			list.add(i);
		}
		final PowerSet<Integer> powerset = new PowerSet<Integer>(list);
		Assert.assertEquals(Math.pow(2, list.size()),
				Double.valueOf(powerset.size()));
	}
	
	@Test
	public void test3() {
		final List<Integer> list = new LinkedList<Integer>();
		for (int i = 0; i < 33; ++i) {
			list.add(i);
		}
		final PowerSet<Integer> powerset = new PowerSet<Integer>(list);
		Assert.assertEquals(Math.pow(2, list.size()),
				Double.valueOf(powerset.size()));
		long counter = 0;
		final Set<List<Integer>> singletons = new HashSet<List<Integer>>(
				ListUtils.map(list,
						new ListUtils.Mapper<Integer, List<Integer>>() {
							
							@Override
							public List<Integer> process(Integer obj) {
								return ListUtils.createSingletonList(obj);
							}
						}));
		for (final List<Integer> subset : powerset) {
			if (subset.size() == 1) {
				singletons.remove(new ArrayList<Integer>(subset));
			}
			counter++;
		}
		Assert.assertTrue(
				String.format("singletons not observed: %s", singletons),
				singletons.isEmpty());
		Assert.assertEquals(Math.pow(2, list.size()), Double.valueOf(counter));
	}
}
